"""Modelli statistici: STL+ETS, ETS, SARIMA e regressione di Fourier con finestra del fermo.

'STL+ETS', 'ETS' e 'SARIMA' si stimano a ogni origine sullo storico (nessun addestramento); se la stima
fallisce o non è finita si ripiega sul profilo a 8 settimane e si conta il ripiego in `storia`.
'Regressione Fourier' è un OLS addestrato in `addestra` su righe (origine, giorno di target),
escluse le origini a livello di fermo (sotto 0.6 volte la mediana dei livelli di addestramento).
"""
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.sarimax import SARIMAX

from base import ModelloBase, livello, storico_pieno
from dati import HORIZON, giorni_da_capodanno

FINESTRA_FERMO = 40
NODI_FERMO = (-13.0, 13.0)    # spline cubica con 5 gradi di libertà su [-40, 40]
COPPIE_FOURIER = 3


def _profilo_8_settimane(origine, storico):
    """Media per giorno della settimana sulle ultime 8 settimane (ripiego)."""
    recente = storico[storico.index >= origine-pd.Timedelta(56, 'D')]
    if len(recente) == 0:
        recente = storico.iloc[-7:]
    per_giorno = recente.groupby(recente.index.dayofweek).mean().reindex(range(7)).fillna(recente.mean())
    giorni = pd.date_range(origine, periods=HORIZON, freq='D')
    return per_giorno.to_numpy()[giorni.dayofweek.to_numpy()]


class _LocaleETS(ModelloBase):
    """Stima un modello locale (ETS, SARIMA) a ogni origine con intervallo all'80%; ripiega sul profilo se fallisce."""

    def __init__(self, nome=None, parametri=None, seed=0):
        super().__init__(nome, parametri, seed)
        self.storia = {'ripiegamenti': 0}

    def addestra(self, dati, fino_a):
        """Nessun addestramento: azzera il conteggio dei ripieghi (uno per utenza nell'esperimento)."""
        self.storia = {'ripiegamenti': 0}

    def _stima(self, y):
        """Restituisce (punto, q10, q90) di lunghezza HORIZON per la serie giornaliera `y`."""
        raise NotImplementedError

    def _una(self, dati, origine):
        storico = storico_pieno(dati, origine)
        if len(storico) == 0:
            raise ValueError(f'{self.nome}: nessun dato prima di {origine:%Y-%m-%d}')
        y = storico.asfreq('D').ffill()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                punto, q10, q90 = (np.asarray(v, dtype=float) for v in self._stima(y))
            if not all(v.shape == (HORIZON,) and np.isfinite(v).all() for v in (punto, q10, q90)):
                raise ValueError('previsione non finita')
        except Exception:
            self.storia['ripiegamenti'] += 1
            punto = _profilo_8_settimane(origine, storico)
            return punto, punto, punto
        return punto, np.minimum(q10, punto), np.maximum(q90, punto)

    def _prevedi(self, dati, origini):
        un = [self._una(dati, o) for o in origini]
        return {'punto': np.stack([u[0] for u in un]), 'q10': np.stack([u[1] for u in un]),
                'q90': np.stack([u[2] for u in un])}


def _da_frame(fit, n):
    quadro = fit.get_prediction(start=n, end=n+HORIZON-1).summary_frame(alpha=0.2)
    colonne = list(quadro.columns)
    inf = [c for c in colonne if 'lower' in c][0]
    sup = [c for c in colonne if 'upper' in c][0]
    return quadro['mean'].to_numpy(), quadro[inf].to_numpy(), quadro[sup].to_numpy()


class StlEts(_LocaleETS):
    """STL (periodo 7) e ETS additivo smorzato sul destagionalizzato."""

    def _stima(self, y):
        fit = STLForecast(y, ETSModel, model_kwargs=dict(error='add', trend='add', damped_trend=True),
                          period=7).fit()
        return _da_frame(fit, len(y))


class Ets(_LocaleETS):
    """ETS con errore e trend additivi smorzati e stagionalità additiva 7."""

    def _stima(self, y):
        fit = ETSModel(y, error='add', trend='add', damped_trend=True, seasonal='add',
                       seasonal_periods=7).fit(disp=False)
        return _da_frame(fit, len(y))


class Sarima(_LocaleETS):
    """SARIMA(1,0,1)(0,1,1)_7: differenza stagionale settimanale, AR e MA di ordine 1.

    Ordini fissi (nessuna selezione automatica per origine); si cambiano con i parametri
    'ordine' e 'ordine_stagionale'.
    """

    def _stima(self, y):
        fit = SARIMAX(y, order=tuple(self.parametri.get('ordine', (1, 0, 1))),
                      seasonal_order=tuple(self.parametri.get('ordine_stagionale', (0, 1, 1, 7)))
                      ).fit(disp=False)
        return _da_frame(fit, len(y))


# --------------------------------------------------------------------------- regressione
def _regressori(giorni):
    """Regressori del solo giorno di target: Fourier annuale, giorno della settimana, finestra del fermo."""
    t = 2*np.pi*giorni.dayofyear.to_numpy()/365.25
    colonne = [np.ones(len(giorni))]
    for k in range(1, COPPIE_FOURIER+1):
        colonne += [np.sin(k*t), np.cos(k*t)]
    settimana = giorni.dayofweek.to_numpy()
    colonne += [(settimana == g).astype(float) for g in range(1, 7)]
    gdc = np.clip(giorni_da_capodanno(giorni), -FINESTRA_FERMO, FINESTRA_FERMO)
    spline = np.asarray(dmatrix('bs(x, knots=nodi, degree=3, lower_bound=-40, upper_bound=40,'
                                ' include_intercept=False) - 1',
                                {'x': gdc, 'nodi': list(NODI_FERMO)}))
    colonne += list(spline.T)
    colonne.append((np.abs(giorni_da_capodanno(giorni)) <= FINESTRA_FERMO).astype(float))
    return np.column_stack(colonne)


class RegressioneFourier(ModelloBase):
    """OLS su y/livello con Fourier annuale, giorno della settimana e finestra del fermo.

    Le origini di addestramento con livello sotto `SOGLIA_LIVELLO` volte la mediana dei livelli di
    addestramento (fermi, ripartenze) sono scartate: con un livello minuscolo i target y/livello
    esploderebbero e dominerebbero l'OLS. In previsione un livello mancante, non positivo o sotto
    la stessa soglia è sostituito dalla mediana, che resta nello stato del modello.
    """

    SOGLIA_LIVELLO = 0.6

    def __init__(self, nome=None, parametri=None, seed=0):
        super().__init__(nome, parametri, seed)
        self._ols = None
        self.mediana_livello = None

    def addestra(self, dati, fino_a):
        fino_a = pd.Timestamp(fino_a)
        serie = dati.serie[dati.serie.index <= fino_a]
        origini = pd.date_range(serie.index[0], fino_a-pd.Timedelta(HORIZON-1, 'D'), freq='D')
        livelli = np.array([livello(dati, o) for o in origini])
        validi = np.isfinite(livelli) & (livelli > 0)
        if not validi.any():
            raise ValueError(f'{self.nome}: nessuna riga di addestramento fino a {fino_a:%Y-%m-%d}')
        mediana = float(np.median(livelli[validi]))
        tenute = validi & (livelli >= self.SOGLIA_LIVELLO*mediana)
        X, y = [], []
        for origine, liv in zip(origini[tenute], livelli[tenute]):
            giorni = pd.date_range(origine, periods=HORIZON, freq='D')
            target = serie.reindex(giorni).to_numpy(dtype=float)/liv
            valido = np.isfinite(target)
            if valido.any():
                X.append(_regressori(giorni)[valido])
                y.append(target[valido])
        if not X:
            raise ValueError(f'{self.nome}: nessuna riga di addestramento fino a {fino_a:%Y-%m-%d}')
        self._ols = sm.OLS(np.concatenate(y), np.vstack(X)).fit()
        self.mediana_livello = mediana
        self.storia = {'mediana_livello': mediana, 'origini_tenute': int(tenute.sum()),
                       'origini_scartate': int(validi.sum()-tenute.sum())}

    def _livello_previsione(self, dati, origine):
        """Livello dell'origine; la mediana di addestramento se è NaN, <= 0 o sotto soglia."""
        liv = livello(dati, origine)
        if not np.isfinite(liv) or liv <= 0 or liv < self.SOGLIA_LIVELLO*self.mediana_livello:
            return self.mediana_livello
        return liv

    def _prevedi(self, dati, origini):
        if self._ols is None:    # senza addestramento esplicito: solo giorni precedenti la prima origine
            self.addestra(dati, origini[0]-pd.Timedelta(1, 'D'))
        punti, q10, q90 = [], [], []
        for origine in origini:
            if len(storico_pieno(dati, origine)) == 0:
                raise ValueError(f'{self.nome}: nessun dato prima di {origine:%Y-%m-%d}')
            liv = self._livello_previsione(dati, origine)
            giorni = pd.date_range(origine, periods=HORIZON, freq='D')
            quadro = self._ols.get_prediction(_regressori(giorni)).summary_frame(alpha=0.2)
            punto = quadro['mean'].to_numpy()
            punti.append(punto*liv)
            q10.append(np.minimum(quadro['obs_ci_lower'].to_numpy(), punto)*liv)
            q90.append(np.maximum(quadro['obs_ci_upper'].to_numpy(), punto)*liv)
        return {'punto': np.stack(punti), 'q10': np.stack(q10), 'q90': np.stack(q90)}


REGISTRO_STATISTICI = {
    'STL+ETS': lambda nome, parametri, seed: StlEts(nome, parametri, seed),
    'ETS': lambda nome, parametri, seed: Ets(nome, parametri, seed),
    'SARIMA': lambda nome, parametri, seed: Sarima(nome, parametri, seed),
    'Regressione Fourier': lambda nome, parametri, seed: RegressioneFourier(nome, parametri, seed),
}

"""Dati EWELD C24 a risoluzione giornaliera per il benchmark a lungo termine.

Caricamento orario (come c24_benchmark), pulizia dei tratti costanti e somma giornaliera.
Unità: kWh. Nessun training all'import.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parent.parent
CARTELLA_C24 = RADICE/'datasets/EWELD/Electricity Consumption/C/C24 Manufacture of basic metals'

HORIZON = 90
TAGLI = (30, 60, 90)
FINE_TRAIN = '2018-09-30'
FINE_VALIDAZIONE = '2019-03-31'
ORIGINI_VALIDAZIONE = pd.date_range('2018-10-01', '2019-01-01', freq='MS')
ORIGINI_TEST = pd.date_range('2019-04-01', '2020-05-01', freq='MS')

FINO_A = {'validazione': FINE_TRAIN, 'test': FINE_VALIDAZIONE}   # ultimo giorno di addestramento per fase

# Capodanno lunare: periodo di fermo noto in anticipo (l'ultima data copre le origini del 2022).
CAPODANNO_LUNARE = ['2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05',
                    '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22']


def carica_oraria(utente):
    """Consumo orario in kWh: somma di quattro misure da 15 minuti, NaN se ne manca una."""
    path = CARTELLA_C24/f'{utente}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Utenza non trovata: {path}')
    d = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(d['Time']))
    if idx.has_duplicates:
        raise ValueError(f'Timestamp duplicati in {path.name}')
    s = pd.Series(pd.to_numeric(d['Value'], errors='raise').to_numpy(), index=idx).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)
    if (s.dropna() < 0).any():
        raise ValueError(f'Consumi negativi in {path.name}')
    serie = s.resample('h').sum(min_count=4).asfreq('h')
    serie.name = utente
    return serie


def pulisci(oraria, ore_min=24):
    """Mette a NaN i tratti di valore costante lunghi almeno `ore_min` ore (zeri o valore bloccato)."""
    gruppi = (oraria != oraria.shift()).cumsum()
    lunghezza = oraria.groupby(gruppi).transform('size')
    return oraria.mask((lunghezza >= ore_min) & oraria.notna())


def a_giornaliera(oraria):
    """Somma giornaliera in kWh: il giorno è NaN se manca anche una sola ora."""
    return oraria.resample('D').sum(min_count=24).asfreq('D')


def carica_giornaliera(utente):
    """Serie giornaliera pulita di un'utenza."""
    serie = a_giornaliera(pulisci(carica_oraria(utente)))
    serie.name = utente
    return serie


def giorni_da_capodanno(index):
    """Giorni (con segno, negativi prima della festa) dal Capodanno lunare più vicino."""
    feste = pd.DatetimeIndex(CAPODANNO_LUNARE)
    differenze = (index.to_numpy()[:, None] - feste.to_numpy()[None, :]) / np.timedelta64(1, 'D')
    vicino = np.abs(differenze).argmin(axis=1)
    return differenze[np.arange(len(index)), vicino]


def calendario(index):
    """Regressori noti in anticipo: giorno della settimana e dell'anno (seno/coseno), weekend, distanza dal Capodanno."""
    settimana = index.dayofweek.to_numpy()
    giorno = index.dayofyear.to_numpy()
    return np.column_stack([np.sin(2*np.pi*settimana/7), np.cos(2*np.pi*settimana/7),
                            np.sin(2*np.pi*giorno/366), np.cos(2*np.pi*giorno/366),
                            settimana >= 5, giorni_da_capodanno(index)]).astype('float32')


@dataclass
class Dati:
    utente: str
    serie: pd.Series        # kWh giornalieri, NaN dove non valido
    scala_mase: float       # MAE del lag 7 sui giorni <= FINE_TRAIN
    origini: dict           # fase -> origini con tutti i 90 giorni di target validi
    scartate: dict          # fase -> origini scartate per target incompleto

    def storico(self, origine):
        """Serie dei soli giorni precedenti l'origine."""
        return self.serie[self.serie.index < origine]

    def target(self, origini):
        """Target (n, HORIZON): i 90 giorni da ciascuna origine; NaN fuori dalla serie."""
        inizi = pd.DatetimeIndex(origini).to_numpy()[:, None]
        giorni = inizi + np.arange(HORIZON).astype('timedelta64[D]')[None, :]
        valori = self.serie.reindex(pd.DatetimeIndex(giorni.ravel())).to_numpy(dtype=float)
        return valori.reshape(len(inizi), HORIZON)

    def fino_a(self, fase):
        """Ultimo giorno utilizzabile per addestrare/selezionare i modelli della fase."""
        return fino_a_fase(fase)


def fino_a_fase(fase):
    """Ultimo giorno di addestramento della fase ('validazione' o 'test'); ValueError altrimenti."""
    if fase not in FINO_A:
        raise ValueError(f'Fase sconosciuta: {fase!r}. Disponibili: {list(FINO_A)}')
    return pd.Timestamp(FINO_A[fase])


def _scala_mase(serie):
    """MAE del lag 7 sui giorni <= FINE_TRAIN (coppie con NaN ignorate)."""
    s = serie[:FINE_TRAIN]
    return float((s - s.shift(7)).abs().mean())


def _dividi_origini(serie, candidate):
    """Separa le origini con tutti i 90 giorni di target validi da quelle scartate."""
    validi, scartate = [], []
    for o in candidate:
        t = serie.reindex(pd.date_range(o, periods=HORIZON, freq='D'))
        (validi if t.notna().all() else scartate).append(o)
    return pd.DatetimeIndex(validi), scartate


def prepara_dati(utente, serie=None):
    """Costruisce `Dati`; se `serie` (giornaliera) è data non legge file (test con serie sintetiche)."""
    if serie is None:
        serie = carica_giornaliera(utente)
    origini, scartate = {}, {}
    for fase, candidate in (('validazione', ORIGINI_VALIDAZIONE), ('test', ORIGINI_TEST)):
        origini[fase], scartate[fase] = _dividi_origini(serie, candidate)
    return Dati(utente, serie, _scala_mase(serie), origini, scartate)


def utenze_idonee():
    """Utenze con almeno 365 giorni validi fino a FINE_VALIDAZIONE e almeno 6 origini di test valide."""
    idonee = []
    for utente in sorted(p.stem for p in CARTELLA_C24.glob('U*.csv')):
        serie = carica_giornaliera(utente)
        giorni = serie[:FINE_VALIDAZIONE].notna().sum()
        if giorni >= 365 and len(_dividi_origini(serie, ORIGINI_TEST)[0]) >= 6:
            idonee.append(utente)
    return idonee

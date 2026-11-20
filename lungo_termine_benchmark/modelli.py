"""Registro dei modelli di previsione a 90 giorni e baseline.

I moduli delle famiglie (statistici, alberi, preaddestrati, reti globali) importano da `base`
e aggiungono i propri costruttori a `REGISTRO`: `costruttore(nome, parametri, seed)`.
"""
import numpy as np
import pandas as pd

from base import ModelloBase, storico_pieno
from dati import HORIZON


def _giorni_target(origine):
    return pd.date_range(origine, periods=HORIZON, freq='D')


# --------------------------------------------------------------------------- baseline
class Baseline(ModelloBase):
    """Baseline senza addestramento: usa solo lo storico precedente l'origine."""

    def _prevedi(self, dati, origini):
        return {'punto': np.stack([self._una(dati, o) for o in origini])}

    def _una(self, dati, origine):
        storico = storico_pieno(dati, origine)
        if len(storico) == 0:
            raise ValueError(f'{self.nome}: nessun dato prima di {origine:%Y-%m-%d}')
        return self._da_storico(dati, origine, storico)


def _naive_settimanale(origine, storico):
    ultimi = storico.to_numpy()[-7:]
    ultimi = np.concatenate([np.full(7-len(ultimi), ultimi[0]), ultimi])
    return np.resize(ultimi, HORIZON)


def _profilo(origine, storico, settimane=8):
    recente = storico[storico.index >= origine-pd.Timedelta(7*settimane, 'D')]
    if len(recente) == 0:
        recente = storico.iloc[-7:]
    per_giorno = recente.groupby(recente.index.dayofweek).mean().reindex(range(7))
    per_giorno = per_giorno.fillna(recente.mean())
    return per_giorno.to_numpy()[_giorni_target(origine).dayofweek.to_numpy()]


class NaiveSettimanale(Baseline):
    """Ultimi 7 giorni osservati ripetuti per 90 giorni."""

    def _da_storico(self, dati, origine, storico):
        return _naive_settimanale(origine, storico)


class Profilo8Settimane(Baseline):
    """Media per giorno della settimana sulle ultime 8 settimane."""

    def _da_storico(self, dati, origine, storico):
        return _profilo(origine, storico)


class AnnoPrecedente(Baseline):
    """Valore di 364 giorni prima, riscalato col rapporto fra le medie degli ultimi 28 giorni.

    Ripiega sul profilo a 8 settimane se manca un valore necessario o la media di riferimento è 0.
    """

    def _da_storico(self, dati, origine, storico):
        grezzo = dati.storico(origine)
        anno = pd.Timedelta(364, 'D')
        ultimi = pd.date_range(origine-pd.Timedelta(28, 'D'), periods=28, freq='D')
        recenti = grezzo.reindex(ultimi).to_numpy(dtype=float)
        riferimento = grezzo.reindex(ultimi-anno).to_numpy(dtype=float)
        valori = grezzo.reindex(_giorni_target(origine)-anno).to_numpy(dtype=float)
        if np.isnan(recenti).any() or np.isnan(riferimento).any() or np.isnan(valori).any() \
                or riferimento.mean() == 0:
            return _profilo(origine, storico)
        return valori*recenti.mean()/riferimento.mean()


REGISTRO = {
    'Naive settimanale': lambda nome, parametri, seed: NaiveSettimanale(nome, parametri, seed),
    'Profilo 8 settimane': lambda nome, parametri, seed: Profilo8Settimane(nome, parametri, seed),
    'Anno precedente': lambda nome, parametri, seed: AnnoPrecedente(nome, parametri, seed),
}
from alberi import REGISTRO_ALBERI  # noqa: E402
from preaddestrati import REGISTRO_PREADDESTRATI  # noqa: E402
from reti import REGISTRO_RETI  # noqa: E402
from statistici import REGISTRO_STATISTICI  # noqa: E402

REGISTRO.update(REGISTRO_STATISTICI)
REGISTRO.update(REGISTRO_ALBERI)
REGISTRO.update(REGISTRO_PREADDESTRATI)
REGISTRO.update(REGISTRO_RETI)
MODELLI = list(REGISTRO)


def crea_modello(nome, parametri=None, seed=0):
    if nome not in REGISTRO:
        raise ValueError(f'Modello sconosciuto: {nome}. Disponibili: {MODELLI}')
    return REGISTRO[nome](nome, parametri, seed)

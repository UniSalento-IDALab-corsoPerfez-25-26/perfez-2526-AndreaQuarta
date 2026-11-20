"""Interfaccia comune dei modelli di previsione a 90 giorni.

Ogni modello espone:
    addestra(dati, fino_a)   usa solo giorni <= fino_a (i modelli globali ricevono la lista di `Dati`)
    prevedi(dati, origini)   dict con 'punto' e opzionali 'q10', 'q90': array (n, HORIZON) in kWh
    storia                   curve di loss (None per i modelli senza addestramento iterativo)

Le sottoclassi implementano `_prevedi`; il troncamento a zero e il controllo delle forme
stanno solo in `prevedi`.
"""
import numpy as np
import pandas as pd

from dati import HORIZON


def storico_pieno(dati, origine):
    """Storico precedente l'origine con i buchi riempiti in avanti (solo valori passati), senza NaN iniziali."""
    return dati.storico(origine).ffill().dropna()


def livello(dati, origine, giorni=28):
    """Media dei `giorni` giorni di calendario precedenti l'origine (storico pieno); NaN se vuoto."""
    piena = storico_pieno(dati, origine)
    ultimi = piena[piena.index >= origine-pd.Timedelta(giorni, 'D')]
    return float(ultimi.mean()) if len(ultimi) else np.nan


class ModelloBase:
    nome = 'ModelloBase'
    globale = False

    def __init__(self, nome=None, parametri=None, seed=0):
        if nome is not None:
            self.nome = nome
        self.parametri = dict(parametri or {})
        self.seed = seed
        self.storia = None

    def addestra(self, dati, fino_a):
        """Nessun addestramento di default (baseline)."""

    def _prevedi(self, dati, origini):
        raise NotImplementedError

    def prevedi(self, dati, origini):
        """Previsioni in kWh, troncate a zero; ogni chiave ha forma (n, HORIZON)."""
        origini = pd.DatetimeIndex(origini)
        uscite = self._prevedi(dati, origini)
        risultato = {}
        for chiave, valori in uscite.items():
            valori = np.asarray(valori, dtype=float)
            if valori.shape != (len(origini), HORIZON):
                raise ValueError(f'{self.nome}: forma {valori.shape} per {chiave!r}, attesa {(len(origini), HORIZON)}')
            risultato[chiave] = np.maximum(valori, 0.0)
        return risultato

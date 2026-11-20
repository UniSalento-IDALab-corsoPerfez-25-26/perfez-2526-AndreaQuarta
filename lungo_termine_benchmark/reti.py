"""Reti globali N-HiTS e PatchTST (neuralforecast), addestrate una volta su tutte le utenze idonee.

Addestramento e previsione girano in sottoprocessi nel venv `.venv_neural` (`_lavoratore_reti.py`,
che non importa nulla da questo progetto): tabelle e previsioni passano da file npz. Il modello
addestrato è salvato con `nf.save` in una cartella temporanea posseduta dall'oggetto (rimossa con
`weakref.finalize`) e ricaricato a ogni `prevedi`.

I NaN della serie non si tolgono: la griglia giornaliera resta regolare, `y` dove mancava è
riempito in avanti con l'ultimo valore passato (0 solo prima del primo valore valido) e
`available_mask` = 0 lo esclude da loss e normalizzazione; il riempimento evita che finestre di
input con lunghi buchi a zero arrivino alla rete. Quantili da `MQLoss(level=[80])`: punto = mediana,
q10/q90 = lo-80/hi-80. N-HiTS usa il calendario come esogene future; PatchTST di neuralforecast non
accetta esogene ed è univariato.

Iperparametri: i default di neuralforecast (nessuna ricerca). A ogni controllo di validazione
(ogni `val_check_steps` epoche) si misura il MAE della mediana in kWh/giorno su `finestre_train`
finestre di train per serie e sulla validazione (ultimi `val_size` giorni di ogni serie, sempre
<= fino_a); si conservano i pesi con il MAE di validazione minimo. `storia` contiene le curve MAE
per epoca (come i grafici di c24_benchmark), l'epoca migliore e la quota di quantili incrociati per utenza.
"""
import json
import os
import shutil
import subprocess
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pandas as pd

from base import ModelloBase
from dati import HORIZON, calendario

RADICE = Path(__file__).resolve().parent.parent
LAVORATORE = Path(__file__).resolve().parent/'_lavoratore_reti.py'
PYTHON_NEURAL = RADICE/'.venv_neural/bin/python'

COLONNE_CALENDARIO = ['sett_sin', 'sett_cos', 'anno_sin', 'anno_cos', 'weekend', 'capodanno']
INPUT_SIZE = 365
CONTESTO = 2*365   # giorni di storico passati a `prevedi` per ogni origine
PREDEFINITI = dict(input_size=INPUT_SIZE, max_steps=1000, val_size=HORIZON, val_check_steps=50,
                   pazienza=5, scaler_type='robust', learning_rate=1e-3, finestre_train=4,
                   thread=4)
EPOCA = pd.Timestamp('1970-01-01')


def _righe(serie, inizio, fine):
    """Griglia giornaliera [inizio, fine] di una serie: y e maschera di disponibilità.

    Dove manca il valore y è l'ultimo valore precedente nella griglia (0 prima del primo valido).
    """
    giorni = pd.date_range(inizio, fine, freq='D')
    valori = serie.reindex(giorni)
    maschera = valori.notna().to_numpy().astype('int8')
    return giorni, valori.ffill().fillna(0.0).to_numpy(dtype=float), maschera


def _tabella(pezzi):
    """pezzi: lista di (unique_id, giorni, y, maschera) -> DataFrame lungo con il calendario."""
    if not pezzi:
        raise ValueError('Nessuna serie con dati validi')
    t = pd.concat([pd.DataFrame({'unique_id': uid, 'ds': giorni, 'y': y, 'available_mask': m})
                   for uid, giorni, y, m in pezzi], ignore_index=True)
    cal = calendario(pd.DatetimeIndex(t['ds']))
    for i, nome in enumerate(COLONNE_CALENDARIO):
        t[nome] = cal[:, i]
    return t


def tabella_addestramento(lista_dati, fino_a):
    """Formato lungo `unique_id, ds, y, available_mask` + calendario; solo `ds <= fino_a`.

    Ogni serie va dal primo all'ultimo giorno valido <= fino_a; i buchi interni restano come righe
    con available_mask = 0 e y riempito in avanti.
    """
    fino_a = pd.Timestamp(fino_a)
    pezzi = []
    for d in lista_dati:
        validi = d.serie[d.serie.index <= fino_a].dropna()
        if len(validi):
            pezzi.append((d.utente, *_righe(d.serie, validi.index[0], validi.index[-1])))
    return _tabella(pezzi)


def tabella_contesti(dati, origini, contesto=CONTESTO):
    """Un contesto per origine (ultimi `contesto` giorni prima dell'origine, fino al giorno precedente)
    con unique_id `utente|AAAA-MM-GG`, e il calendario dei HORIZON giorni futuri."""
    pezzi, futuri = [], []
    for o in origini:
        validi = dati.storico(o).dropna()
        validi = validi[validi.index >= o-pd.Timedelta(contesto, 'D')]
        if len(validi) == 0:
            raise ValueError(f'{dati.utente}: nessun dato prima di {o:%Y-%m-%d}')
        uid = f'{dati.utente}|{o:%Y-%m-%d}'
        pezzi.append((uid, *_righe(dati.serie, validi.index[0], o-pd.Timedelta(1, 'D'))))
        futuri.append((uid, pd.date_range(o, periods=HORIZON, freq='D')))
    futuro = pd.DataFrame({'unique_id': np.repeat([u for u, _ in futuri], HORIZON),
                           'ds': np.concatenate([g.to_numpy() for _, g in futuri])})
    return _tabella(pezzi), futuro


def ordina_quantili(q10, punto, q90):
    """MQLoss non impone l'ordine dei quantili: il punto (mediana) resta, q10/q90 lo includono.

    Restituisce le previsioni e la quota di punti incrociati (mediana fuori da [q10, q90] o q10 > q90).
    """
    incrociati = (punto < q10) | (punto > q90) | (q10 > q90)
    return ({'punto': punto, 'q10': np.minimum(q10, punto), 'q90': np.maximum(q90, punto)},
            float(incrociati.mean()))


def _giorni(ds):
    return ((pd.DatetimeIndex(ds)-EPOCA)//pd.Timedelta(1, 'D')).to_numpy(dtype='int64')


def _array(t):
    return {'unique_id': t['unique_id'].to_numpy(dtype=str), 'ds': _giorni(t['ds']),
            'y': t['y'].to_numpy(dtype='float32'), 'available_mask': t['available_mask'].to_numpy(dtype='int8'),
            'esogene': t[COLONNE_CALENDARIO].to_numpy(dtype='float32')}


def _esegui_processo(richiesta, array, python=PYTHON_NEURAL):
    """Lancia il lavoratore; restituisce (output.npz come dict o None, storia.json o None)."""
    if not Path(python).is_file():
        raise FileNotFoundError(f"{richiesta['nome']}: venv mancante {python}")
    with tempfile.TemporaryDirectory(prefix='lt_reti_') as cartella:
        cartella = Path(cartella)
        (cartella/'richiesta.json').write_text(json.dumps(richiesta))
        np.savez(cartella/'input.npz', **array)
        env = {**os.environ}
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        processo = subprocess.run([str(python), str(LAVORATORE), str(cartella)],
                                  env=env, capture_output=True, text=True)
        if processo.returncode != 0:
            raise RuntimeError(f"{richiesta['nome']} ({richiesta['operazione']}): processo terminato "
                               f'con codice {processo.returncode}\n{processo.stderr[-3000:]}')
        uscita = None
        if (cartella/'output.npz').is_file():
            with np.load(cartella/'output.npz') as out:
                uscita = {k: out[k] for k in out.files}
        storia = None
        if (cartella/'storia.json').is_file():
            storia = json.loads((cartella/'storia.json').read_text())
        return uscita, storia


class RetiGlobale(ModelloBase):
    """Rete neuralforecast globale: `addestra` su tutte le serie, `prevedi` per una utenza."""

    globale = True

    def __init__(self, nome=None, parametri=None, seed=0):
        super().__init__(nome, parametri, seed)
        self.config = {**PREDEFINITI, **self.parametri}
        self.cartella_modello = None
        self.fino_a = None

    def _richiesta(self, operazione):
        return dict(operazione=operazione, nome=self.nome, horizon=HORIZON, seed=int(self.seed),
                    esogene=COLONNE_CALENDARIO, cartella_modello=self.cartella_modello, **self.config)

    def addestra(self, dati, fino_a):
        """Addestra su tutte le serie della lista (o su una sola `Dati`) con ds <= fino_a."""
        lista = dati if isinstance(dati, (list, tuple)) else [dati]
        if self.cartella_modello is None:
            self.cartella_modello = tempfile.mkdtemp(prefix='lt_rete_')
            weakref.finalize(self, shutil.rmtree, self.cartella_modello, ignore_errors=True)
        self.fino_a = pd.Timestamp(fino_a)
        tabella = tabella_addestramento(lista, self.fino_a)
        _, self.storia = _esegui_processo(self._richiesta('addestra'), _array(tabella))

    def _prevedi(self, dati, origini):
        if self.fino_a is None:
            raise RuntimeError(f'{self.nome}: modello non addestrato')
        storico, futuro = tabella_contesti(dati, origini)
        array = _array(storico)
        array.update(futuro_unique_id=futuro['unique_id'].to_numpy(dtype=str),
                     futuro_ds=_giorni(futuro['ds']),
                     futuro_esogene=calendario(pd.DatetimeIndex(futuro['ds'])))
        richiesta = self._richiesta('prevedi')
        richiesta['ordine'] = [f'{dati.utente}|{o:%Y-%m-%d}' for o in origini]
        uscita, _ = _esegui_processo(richiesta, array)
        previsione, quota = ordina_quantili(uscita['q10'], uscita['punto'], uscita['q90'])
        self.storia.setdefault('quantili_incrociati', {})[dati.utente] = quota
        return previsione


RETI = ['N-HiTS', 'PatchTST']
REGISTRO_RETI = {nome: (lambda nome, parametri, seed: RetiGlobale(nome, parametri, seed)) for nome in RETI}

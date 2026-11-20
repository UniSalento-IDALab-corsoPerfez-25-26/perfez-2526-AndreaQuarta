"""Alberi diretti (LightGBM, XGBoost): un solo regressore con l'orizzonte come feature.

Una riga per (origine, giorno di target); il target è il consumo diviso per la media dei 28 giorni
precedenti l'origine, e la previsione viene riportata in kWh moltiplicando per lo stesso livello.
L'addestramento e la previsione girano in un sottoprocesso (questo stesso file) per evitare
conflitti OpenMP con PyTorch; qui si importa solo da `base` e `dati`.
"""
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from base import ModelloBase, livello
from dati import HORIZON, calendario

# Punto di partenza: gli iperparametri di c24_benchmark/modelli.py::PARAMETRI, ma numero di alberi fisso
# (nessun early stopping): lo sceglie il notebook sulla fase di validazione variando `parametri`.
PARAMETRI = {
    'XGBoost': dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                    colsample_bytree=0.5, min_child_weight=5),
    'LightGBM': dict(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.5, min_child_samples=20),
}
THREAD = 4

# Colonne: orizzonte, calendario (6), media 7 giorni / livello, valore di 364 giorni prima / livello.
COLONNA_ORIZZONTE = 0
COLONNA_MEDIA_7 = 7
COLONNA_ANNO_PRIMA = 8
N_CARATTERISTICHE = 9


def caratteristiche(dati, origini):
    """(X, y) con una riga per (origine, giorno di target), in ordine origine-maggiore.

    Con livello (media ultimi 28 giorni) assente o <= 0 le colonne normalizzate e y sono NaN.
    Usa solo giorni precedenti l'origine; y è NaN se manca il consumo del giorno di target.
    """
    x, y, _ = _costruisci(dati, origini)
    return x, y


def _costruisci(dati, origini):
    """Come `caratteristiche`, più il livello (media 28 giorni) di ogni origine."""
    origini = pd.DatetimeIndex(origini)
    l28 = np.array([livello(dati, o, 28) for o in origini])
    l7 = np.array([livello(dati, o, 7) for o in origini])
    anno = pd.Timedelta(364, 'D')
    blocchi, bersagli = [], []
    for o, liv, media7 in zip(origini, l28, l7):
        giorni = pd.date_range(o, periods=HORIZON, freq='D')
        x = np.full((HORIZON, N_CARATTERISTICHE), np.nan)
        x[:, COLONNA_ORIZZONTE] = np.arange(1, HORIZON+1)
        x[:, 1:7] = calendario(giorni)
        passato = giorni-anno
        anno_prima = dati.serie.reindex(passato).to_numpy(dtype=float)
        anno_prima[passato >= o] = np.nan
        y = dati.serie.reindex(giorni).to_numpy(dtype=float)
        if np.isfinite(liv) and liv > 0:
            x[:, COLONNA_MEDIA_7] = media7/liv
            x[:, COLONNA_ANNO_PRIMA] = anno_prima/liv
            y = y/liv
        else:
            y = np.full(HORIZON, np.nan)
        blocchi.append(x)
        bersagli.append(y)
    if not blocchi:
        return np.empty((0, N_CARATTERISTICHE)), np.empty(0), l28
    return np.vstack(blocchi), np.concatenate(bersagli), l28


class Albero(ModelloBase):
    """Regressore diretto (orizzonte come feature) addestrato in un sottoprocesso."""

    def __init__(self, nome=None, parametri=None, seed=0):
        super().__init__(nome, parametri, seed)
        self._modello = None        # byte del modello serializzato

    def _iperparametri(self):
        p = {**PARAMETRI[self.nome], **self.parametri}
        return p, int(p.pop('thread', THREAD))

    def addestra(self, dati, fino_a):
        fino_a = pd.Timestamp(fino_a)
        serie = dati.serie[dati.serie.index <= fino_a]
        if len(serie) == 0:
            raise ValueError(f'{self.nome}: nessun dato fino a {fino_a:%Y-%m-%d}')
        ristretti = replace(dati, serie=serie)
        origini = pd.date_range(serie.index[0], fino_a-pd.Timedelta(HORIZON-1, 'D'), freq='D')
        x, y = caratteristiche(ristretti, origini)
        valida = np.isfinite(y)
        if not valida.any():
            raise ValueError(f'{self.nome}: righe di addestramento insufficienti fino a {fino_a:%Y-%m-%d}')
        parametri, thread = self._iperparametri()
        richiesta = dict(nome=self.nome, parametri=parametri, seed=self.seed, thread=thread)
        risultato = _esegui_processo_alberi(richiesta, dict(x_train=x[valida], y_train=y[valida]), thread)
        self._modello = risultato['modello']
        self.storia = {'alberi': int(parametri['n_estimators']), 'loss_train': float(risultato['loss_train'])}
        return self

    def _prevedi(self, dati, origini):
        if self._modello is None:   # senza addestramento esplicito: solo giorni precedenti la prima origine
            self.addestra(dati, origini[0]-pd.Timedelta(1, 'D'))
        x, _, l28 = _costruisci(dati, origini)
        if np.isnan(l28).any():
            raise ValueError(f'{self.nome}: nessun dato prima di {origini[np.isnan(l28)][0]:%Y-%m-%d}')
        _, thread = self._iperparametri()
        richiesta = dict(nome=self.nome, thread=thread)
        z = _esegui_processo_alberi(richiesta, dict(x=x, modello=self._modello), thread, '--prevedi')['z']
        livello = np.where(np.isfinite(l28), l28, 0.0)
        return {'punto': z.reshape(len(origini), HORIZON)*livello[:, None]}


def _esegui_processo_alberi(richiesta, array, thread, modalita='--addestra'):
    """Lancia questo file con lo stesso interprete; scambio dati tramite npz in una cartella temporanea."""
    env = {**os.environ, **{k: str(thread) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                       'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}}
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    with tempfile.TemporaryDirectory(prefix='lt_') as cartella:
        cartella = Path(cartella)
        (cartella/'richiesta.json').write_text(json.dumps(richiesta))
        np.savez(cartella/'input.npz', **array)
        processo = subprocess.run([sys.executable, str(Path(__file__).resolve()), modalita, str(cartella)],
                                  env=env, capture_output=True, text=True)
        if processo.returncode != 0:
            raise RuntimeError(f"{richiesta['nome']}: processo terminato con codice {processo.returncode}\n"
                               f'{processo.stderr[-3000:]}')
        with np.load(cartella/'output.npz') as out:
            return {k: out[k] for k in out.files}


def _lavoratore_addestra(cartella):
    """Eseguito solo nel processo separato."""
    cartella = Path(cartella)
    r = json.loads((cartella/'richiesta.json').read_text())
    with np.load(cartella/'input.npz') as d:
        xtr, ytr = d['x_train'], d['y_train']
    p = dict(r['parametri'])
    if r['nome'] == 'XGBoost':
        import xgboost as xgb
        m = xgb.XGBRegressor(**p, tree_method='hist', eval_metric='rmse', n_jobs=r['thread'],
                             random_state=r['seed'])
        m.fit(xtr, ytr, eval_set=[(xtr, ytr)], verbose=False)
        perdita = float(m.evals_result()['validation_0']['rmse'][-1])**2
        modello = np.frombuffer(bytes(m.get_booster().save_raw()), dtype=np.uint8)
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=r['thread'], random_state=r['seed'],
                              deterministic=True, force_row_wise=True)
        registro = {}
        m.fit(xtr, ytr, eval_set=[(xtr, ytr)], eval_names=['train'], eval_metric='l2',
              callbacks=[lgb.record_evaluation(registro)])
        perdita = float(registro['train']['l2'][-1])
        modello = np.frombuffer(m.booster_.model_to_string().encode(), dtype=np.uint8)
    np.savez(cartella/'output.npz', loss_train=perdita, modello=modello)


def _lavoratore_prevedi(cartella):
    cartella = Path(cartella)
    r = json.loads((cartella/'richiesta.json').read_text())
    with np.load(cartella/'input.npz') as d:
        x, modello = d['x'], d['modello'].tobytes()
    if r['nome'] == 'XGBoost':
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(bytearray(modello))
        z = b.inplace_predict(x, validate_features=False)
    else:
        import lightgbm as lgb
        b = lgb.Booster(model_str=modello.decode())
        z = b.predict(x, num_threads=r['thread'])
    np.savez(cartella/'output.npz', z=np.asarray(z, dtype=float))


REGISTRO_ALBERI = {
    'LightGBM': lambda nome, parametri, seed: Albero(nome, parametri, seed),
    'XGBoost': lambda nome, parametri, seed: Albero(nome, parametri, seed),
}

if __name__ == '__main__' and len(sys.argv) == 3:
    {'--addestra': _lavoratore_addestra, '--prevedi': _lavoratore_prevedi}[sys.argv[1]](sys.argv[2])

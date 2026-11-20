"""Esperimento: addestra e valuta i modelli su una o più utenze e salva i risultati.

I modelli globali sono addestrati una sola volta su tutte le utenze idonee; gli altri per utente.
Un errore di un (modello, utente) viene registrato e il giro continua: anche un errore nella
preparazione dei dati di un'utenza o nella creazione di un modello.
"""
import copy
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dati import (FINE_TRAIN, FINE_VALIDAZIONE, HORIZON, ORIGINI_TEST, ORIGINI_VALIDAZIONE, TAGLI, fino_a_fase,
                  prepara_dati, utenze_idonee)
from modelli import crea_modello

CARTELLA = Path(__file__).resolve().parent
sys.path.append(str(CARTELLA.parent))   # metriche.py, unico per tutti i benchmark, sta nella radice
import metriche  # noqa: E402

CARTELLA_RISULTATI = CARTELLA/'risultati'
FERMO_DA, FERMO_A = pd.Timestamp('2019-11-01'), pd.Timestamp('2020-02-01')
RIFERIMENTO_DM = 'Naive settimanale'    # Diebold-Mariano di ogni modello contro questo, per utente
PACCHETTI = {
    'principale': ['numpy', 'pandas', 'statsmodels', 'patsy', 'lightgbm', 'xgboost'],
    '.venv_neural': ['neuralforecast', 'pytorch_lightning', 'torch', 'numpy'],
    '.venv_chronos': ['chronos_forecasting', 'torch', 'numpy'],
    '.venv_timesfm': ['timesfm', 'mlx', 'torch', 'numpy'],
    '.venv_timestral': ['timestral', 'torch', 'numpy'],
}


def _versioni():
    """Versioni dei pacchetti principali: di questo interprete e dei venv dei sottoprocessi (dai dist-info)."""
    versioni = {'principale': {'python': platform.python_version()}}
    for nome in PACCHETTI['principale']:
        try:
            versioni['principale'][nome] = importlib.metadata.version(nome)
        except importlib.metadata.PackageNotFoundError:
            versioni['principale'][nome] = None
    for venv, pacchetti in PACCHETTI.items():
        if venv == 'principale':
            continue
        versioni[venv] = {}
        for nome in pacchetti:
            info = sorted((CARTELLA.parent/venv).glob(f'lib/python*/site-packages/{nome}-*.dist-info'))
            versioni[venv][nome] = info[-1].name[len(nome)+1:-len('.dist-info')] if info else None
    return versioni


def _commit():
    """Hash del commit corrente e presenza di modifiche non registrate (None fuori da git)."""
    try:
        def git(*argomenti):
            return subprocess.run(['git', *argomenti], cwd=CARTELLA, capture_output=True, text=True,
                                  check=True).stdout.strip()
        return git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--untracked-files=no'))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _json(valore):
    """Conversione per json.dumps dei tipi numpy/pandas presenti in `storia`."""
    if isinstance(valore, np.generic):
        return valore.item()
    if isinstance(valore, np.ndarray):
        return valore.tolist()
    if isinstance(valore, pd.Timestamp):
        return f'{valore:%Y-%m-%d}'
    raise TypeError(f'Tipo non serializzabile: {type(valore).__name__}')


def _righe_previsioni(utente, nome, origini, reale, previsione):
    n = len(origini)
    giorni = (np.asarray(origini, dtype='datetime64[D]')[:, None]
              + np.arange(HORIZON).astype('timedelta64[D]')[None, :])
    nan = np.full((n, HORIZON), np.nan)
    return pd.DataFrame({
        'utente': utente, 'modello': nome,
        'origine': np.repeat(np.asarray(origini), HORIZON),
        'giorno': giorni.ravel(), 'reale': reale.ravel(),
        'punto': previsione['punto'].ravel(),
        'q10': previsione.get('q10', nan).ravel(), 'q90': previsione.get('q90', nan).ravel()})


def _metriche(utente, nome, reale, previsione, scala):
    tab = metriche.tabella({nome: previsione}, reale, scala, TAGLI)
    tab.index = pd.MultiIndex.from_tuples([(utente, nome, t) for _, t in tab.index],
                                          names=['utente', 'modello', 'taglio'])
    return tab


def _confronti_dm(per_utente):
    """per_utente: {utente: (reale, {modello: previsione})}; None se il riferimento manca ovunque."""
    tabelle = []
    for u, (reale, previsioni) in per_utente.items():
        if RIFERIMENTO_DM not in previsioni or len(previsioni) < 2:
            continue
        tab = metriche.confronto_dm(previsioni, reale, RIFERIMENTO_DM, TAGLI)
        tab.index = pd.MultiIndex.from_tuples([(u, *i) for i in tab.index], names=['utente', 'modello', 'taglio'])
        tabelle.append(tab)
    return pd.concat(tabelle) if tabelle else None


def _errore(e):
    return f'{type(e).__name__}: {e}'


def esegui(utenti, modelli, fase='test', parametri=None, cartella=None):
    """Esegue il giro `modelli` x `utenti` sulla fase ('validazione' o 'test') e salva i risultati."""
    parametri = parametri or {}
    fino_a = fino_a_fase(fase)
    inizio_giro = datetime.now()
    dati, errori_dati = {}, {}
    for u in utenti:
        try:
            dati[u] = prepara_dati(u)
        except Exception as e:  # noqa: BLE001 - l'utenza resta fuori, l'errore va per ogni modello
            errori_dati[u] = _errore(e)
            traceback.print_exc()
    globali = None
    righe_prev, tabelle, tabelle_fermo, errori, secondi, storie = [], [], [], [], {}, {}
    per_utente = {}

    for nome in modelli:
        t0 = time.perf_counter()
        try:
            modello = crea_modello(nome, parametri.get(nome))
            if modello.globale:
                if globali is None:
                    globali = [prepara_dati(u) for u in utenze_idonee()]
                modello.addestra(globali, fino_a)
        except Exception as e:  # noqa: BLE001 - l'errore va registrato, il giro continua
            errori += [{'utente': u, 'modello': nome, 'errore': _errore(e)} for u in utenti]
            traceback.print_exc()
            secondi[nome] = time.perf_counter()-t0
            continue
        storie[nome] = {}
        for u in utenti:
            if u in errori_dati:
                errori.append({'utente': u, 'modello': nome, 'errore': errori_dati[u]})
                continue
            d = dati[u]
            origini = d.origini[fase]
            if len(origini) == 0:
                continue
            try:
                if not modello.globale:
                    modello.addestra(d, fino_a)
                previsione = modello.prevedi(d, origini)
                reale = d.target(origini)
                tabelle.append(_metriche(u, nome, reale, previsione, d.scala_mase))
                righe_prev.append(_righe_previsioni(u, nome, origini, reale, previsione))
                per_utente.setdefault(u, (reale, {}))[1][nome] = previsione
                if fase == 'test':
                    f = np.asarray((origini >= FERMO_DA) & (origini <= FERMO_A))
                    if f.any():
                        tabelle_fermo.append(_metriche(u, nome, reale[f], {k: v[f] for k, v in previsione.items()},
                                                       d.scala_mase))
                if not modello.globale:
                    storie[nome][u] = copy.deepcopy(modello.storia)
            except Exception as e:  # noqa: BLE001
                errori.append({'utente': u, 'modello': nome, 'errore': _errore(e)})
                traceback.print_exc()
        if modello.globale:
            storie[nome] = {'globale': copy.deepcopy(modello.storia)}
        secondi[nome] = time.perf_counter()-t0

    colonne = ['utente', 'modello', 'origine', 'giorno', 'reale', 'punto', 'q10', 'q90']
    risultato = {
        'metriche': pd.concat(tabelle) if tabelle else pd.DataFrame(columns=metriche.ORDINE),
        'previsioni': pd.concat(righe_prev, ignore_index=True) if righe_prev else pd.DataFrame(columns=colonne),
        'errori': pd.DataFrame(errori, columns=['utente', 'modello', 'errore']),
        'metriche_fermo': pd.concat(tabelle_fermo) if tabelle_fermo else None,
        'dm': _confronti_dm(per_utente),
        'secondi': secondi,
        'storia': storie,
    }

    if cartella is None:
        etichetta = utenti[0] if len(utenti) == 1 else 'multi'
        cartella = CARTELLA_RISULTATI/f'{etichetta}_{inizio_giro:%Y%m%d_%H%M%S}'
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    risultato['cartella'] = cartella
    commit, modifiche = _commit()
    configurazione = {
        'utenti': list(utenti), 'modelli': list(modelli), 'fase': fase, 'fino_a': f'{fino_a:%Y-%m-%d}',
        'fine_train': FINE_TRAIN, 'fine_validazione': FINE_VALIDAZIONE, 'horizon': HORIZON,
        'origini_validazione': [f'{o:%Y-%m-%d}' for o in ORIGINI_VALIDAZIONE],
        'origini_test': [f'{o:%Y-%m-%d}' for o in ORIGINI_TEST],
        'parametri': parametri,
        'origini_scartate': {u: [f'{o:%Y-%m-%d}' for o in d.scartate[fase]] for u, d in dati.items()},
        'secondi': secondi,
        'storia': storie,
        'commit': commit, 'modifiche_non_registrate': modifiche,
        'versioni': _versioni(),
    }
    (cartella/'configurazione.json').write_text(json.dumps(configurazione, indent=2, ensure_ascii=False,
                                                           default=_json))
    risultato['metriche'].to_csv(cartella/'metriche.csv')
    risultato['previsioni'].to_csv(cartella/'previsioni.csv', index=False)
    risultato['errori'].to_csv(cartella/'errori.csv', index=False)
    if risultato['metriche_fermo'] is not None:
        risultato['metriche_fermo'].to_csv(cartella/'metriche_fermo.csv')
    if risultato['dm'] is not None:
        risultato['dm'].to_csv(cartella/'dm.csv')
    return risultato

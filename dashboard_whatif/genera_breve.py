"""Rilancia il benchmark orario c24 (U141) con orizzonte 96h e salva le previsioni dirette
per la dashboard. Non modifica c24_benchmark: ne importa solo dati.py e modelli.py.

Uso (dalla radice del progetto):  .venv_eweld/bin/python dashboard_whatif/genera_breve.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RADICE/'c24_benchmark'), str(RADICE)]
import dati      # noqa: E402
import modelli   # noqa: E402

UTENTE = 'U141'
PERIODI = {'train': ('2018-11-01', '2019-09-30'),
           'validazione': ('2019-10-01', '2019-10-31'),
           'test': ('2019-11-01', '2019-12-31')}
HORIZON = 96
LOOKBACK = 168
MODELLI = ['Persistenza', 'Naive 24h', 'Naive 168h', 'XGBoost', 'LightGBM', 'LSTM', 'GRU',
           'Chronos-2', 'Chronos-Bolt', 'TimesFM 2.5', 'TimesFM 3.0']
USCITA = Path(__file__).resolve().parent/'dati'/'breve'

D = dati.prepara_dati(UTENTE, PERIODI, horizon=HORIZON, lookback=LOOKBACK)
print(f'{len(D.test)} origini di test, scala MASE {D.scala_mase:.1f}', flush=True)

ore = np.arange(1, HORIZON+1)
righe = []
for nome in MODELLI:
    inizio = time.perf_counter()
    m = modelli.crea_modello(nome, seed=0)
    m.addestra(D, None)
    p = np.asarray(m.prevedi_test(D)['diretta'])
    print(f'{nome}: {time.perf_counter()-inizio:.0f}s', flush=True)
    for i, o in enumerate(D.test.origini):
        righe.append(pd.DataFrame({'modello': nome, 'origine': o, 'ora': ore,
                                   'istante': o+pd.to_timedelta(ore-1, 'h'),
                                   'reale': D.test.target[i], 'punto': p[i]}))

USCITA.mkdir(parents=True, exist_ok=True)
pd.concat(righe).to_csv(USCITA/'previsioni.csv', index=False)
D.serie.loc[PERIODI['validazione'][0]:PERIODI['test'][1]].rename('kwh').to_csv(USCITA/'serie.csv')
(USCITA/'configurazione.json').write_text(json.dumps(dict(
    utente=UTENTE, periodi=PERIODI, horizon=HORIZON, lookback=LOOKBACK,
    scala_mase=float(D.scala_mase), modelli=MODELLI), indent=2))
print('Salvato in', USCITA)

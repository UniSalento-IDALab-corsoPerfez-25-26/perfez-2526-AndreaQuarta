"""Copia l'ultimo run U141 del benchmark a medio termine e ci aggiunge la serie giornaliera
pulita (per lo storico nel grafico). Non riaddestra niente.

Uso (dalla radice del progetto):  .venv_eweld/bin/python dashboard_whatif/genera_lungo.py
"""
import json
import shutil
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RADICE/'lungo_termine_benchmark'), str(RADICE)]
import dati  # noqa: E402

UTENTE = 'U141'
# solo i run completi (U141_AAAAMMGG_hhmmss): esclude le prove parziali come U141_timestral_*
RUN = sorted((RADICE/'lungo_termine_benchmark'/'risultati').glob(f'{UTENTE}_[0-9]*'))[-1]
USCITA = Path(__file__).resolve().parent/'dati'/'lungo'
USCITA.mkdir(parents=True, exist_ok=True)

for f in ('previsioni.csv', 'metriche.csv'):
    shutil.copy(RUN/f, USCITA/f)
D = dati.prepara_dati(UTENTE)
D.serie.rename('kwh').to_csv(USCITA/'serie.csv')
(USCITA/'configurazione.json').write_text(json.dumps(dict(
    utente=UTENTE, run=RUN.name, scala_mase=float(D.scala_mase)), indent=2))
print('Run', RUN.name, '->', USCITA)

"""Modelli preaddestrati zero-shot (Chronos-2, Chronos-Bolt, TimesFM 2.5 e 3.0) su serie giornaliere.

I modelli girano in un sottoprocesso nel proprio venv (`_lavoratore_preaddestrati.py`), che non
importa nulla da questo progetto: contesti e previsioni passano da file npz. Nessun addestramento.
Il contesto sono gli ultimi `contesto` giorni precedenti l'origine: per Chronos-2 e Chronos-Bolt la
serie grezza (i NaN interni restano, Chronos li maschera; tolti solo i NaN iniziali), per TimesFM lo
storico pieno (buchi riempiti in avanti), perché TimesFM non accetta valori mancanti.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from base import ModelloBase, storico_pieno
from dati import HORIZON

RADICE = Path(__file__).resolve().parent.parent
LAVORATORE = Path(__file__).resolve().parent/'_lavoratore_preaddestrati.py'

PREADDESTRATI = {
    'Chronos-2':    dict(venv='.venv_chronos', pesi='~/.cache/eweld_chronos/chronos-2'),
    'Chronos-Bolt': dict(venv='.venv_chronos', pesi='~/.cache/eweld_chronos/chronos-bolt-base'),
    'TimesFM 2.5':  dict(venv='.venv_timesfm', pesi='~/.cache/eweld_timesfm/timesfm-2.5-200m-pytorch'),
    'TimesFM 3.0':  dict(venv='.venv_timesfm', pesi='~/.cache/eweld_timesfm/timesfm-3.0-pytorch'),
}
# Timestral (Mistral, ottobre 2026): registrato solo se il suo venv esiste, così i giri senza di lui non cambiano.
# Preparazione: venv `.venv_timestral` con la libreria `timestral` e pesi in ~/.cache/eweld_timestral/.
if (RADICE/'.venv_timestral/bin/python').is_file():
    PREADDESTRATI['Timestral 209M'] = dict(venv='.venv_timestral', pesi='~/.cache/eweld_timestral/timestral-209m')
CONTESTO = 2048   # giorni di storico in ingresso


def contesto_modello(nome, dati, origine, contesto=CONTESTO):
    """Contesto del modello `nome` per un'origine: solo giorni < origine, al più `contesto` giorni."""
    if nome.startswith(('Chronos', 'Timestral')):   # accettano NaN come valore mancante
        grezzo = dati.storico(origine).to_numpy(dtype=float)[-contesto:]
        validi = np.flatnonzero(np.isfinite(grezzo))
        return grezzo[validi[0]:] if len(validi) else grezzo[:0]
    return storico_pieno(dati, origine).to_numpy(dtype=float)[-contesto:]


def impacchetta(contesti):
    """Serie di lunghezza diversa -> {'valori': (n, max) riempito con NaN a destra, 'lunghezze': (n,)}.

    I NaN dentro una serie (Chronos) restano: `lunghezze` separa il riempimento dai valori reali.
    """
    lunghezze = np.array([len(c) for c in contesti], dtype=int)
    valori = np.full((len(contesti), lunghezze.max()), np.nan)
    for i, c in enumerate(contesti):
        valori[i, :len(c)] = c
    return {'valori': valori, 'lunghezze': lunghezze}


def spacchetta(valori, lunghezze):
    """Inverso di `impacchetta`: lista di array float32 senza il riempimento."""
    return [np.asarray(valori[i, :n], dtype=np.float32) for i, n in enumerate(lunghezze)]


def _esegui_processo(richiesta, array, python):
    """Lancia il lavoratore con l'interprete `python`; scambio dati tramite npz in una cartella temporanea."""
    with tempfile.TemporaryDirectory(prefix='lt_') as cartella:
        cartella = Path(cartella)
        (cartella/'richiesta.json').write_text(json.dumps(richiesta))
        np.savez(cartella/'input.npz', **array)
        env = {**os.environ}
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        processo = subprocess.run([str(python), str(LAVORATORE), str(cartella)],
                                  env=env, capture_output=True, text=True)
        if processo.returncode != 0:
            raise RuntimeError(f"{richiesta['nome']}: processo terminato con codice {processo.returncode}\n"
                               f'{processo.stderr[-3000:]}')
        with np.load(cartella/'output.npz') as out:
            return {k: out[k] for k in out.files}


class Preaddestrato(ModelloBase):
    """Previsione zero-shot: un sottoprocesso per chiamata, tutte le origini in un solo batch."""

    def _prevedi(self, dati, origini):
        info = PREADDESTRATI[self.nome]
        python = RADICE/info['venv']/'bin/python'
        if not python.is_file():
            raise FileNotFoundError(f'{self.nome}: venv mancante {python}')
        contesto = int(self.parametri.get('contesto', CONTESTO))
        contesti = []
        for o in origini:
            storico = contesto_modello(self.nome, dati, o, contesto)
            if len(storico) == 0:
                raise ValueError(f'{self.nome}: nessun dato prima di {o:%Y-%m-%d}')
            contesti.append(storico)
        richiesta = dict(nome=self.nome, pesi=os.path.expanduser(info['pesi']),
                         horizon=HORIZON, contesto=contesto)
        return _esegui_processo(richiesta, impacchetta(contesti), python)


REGISTRO_PREADDESTRATI = {nome: (lambda nome, parametri, seed: Preaddestrato(nome, parametri, seed))
                          for nome in PREADDESTRATI}

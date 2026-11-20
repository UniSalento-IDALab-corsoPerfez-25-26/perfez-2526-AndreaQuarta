"""Algoritmi di previsione a 48 ore con interfaccia comune.

Ogni modello espone:
    addestra(dati, grafico=None)  usa solo train e validazione
    prevedi_test(dati)            previsioni in kWh sul test nelle quattro modalità (vedi VALUTAZIONI)
    storia                        curve di loss su train e validazione (None per i baseline)

Addestramento: strategia diretta multi-output, le H ore in un solo passo dall'origine.
Lo stesso modello addestrato viene poi valutato in quattro modi (nessun nuovo training):
    diretta       a ogni origine (ogni H ore) prevede +1…+H usando solo lo storico reale
    breve         a ogni ora di test prevede +1…+ore_breve con lo storico reale aggiornato ogni ora
                  (prime ore_breve uscite della stessa previsione diretta)
    one_step      a ogni ora di test prevede +1h con lo storico reale aggiornato ogni ora
    ricorsiva     a ogni origine prevede +1h, reinserisce la propria previsione nello storico
                  e ripete H volte (autoregressiva: dopo la prima ora non vede valori reali)

Le loss sono MSE sul target standardizzato (media e deviazione del train), quindi
confrontabili fra modelli. XGBoost e LightGBM girano in un processo separato:
nello stesso processo di PyTorch il runtime OpenMP su macOS va in crash o si blocca.

I modelli preaddestrati (Chronos, TimesFM) sono zero-shot: nessun addestramento, girano
nel proprio venv tramite questo stesso file. Per questo a livello di modulo si importa
solo numpy (il venv di TimesFM non ha pandas).
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

RADICE = Path(__file__).resolve().parent.parent

PARAMETRI = {
    'XGBoost': dict(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                    colsample_bytree=0.5, min_child_weight=5, early_stopping_rounds=50),
    'LightGBM': dict(n_estimators=600, learning_rate=0.05, num_leaves=15, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.5, min_child_samples=20),
    'LSTM': dict(hidden=64, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch=64, epoche=60, pazienza=8),
    'GRU': dict(hidden=64, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch=64, epoche=60, pazienza=8),
}

# Zero-shot: venv, cartella dei pesi (già scaricati in locale, nessun accesso a internet).
PREADDESTRATI = {
    'Chronos-2':   dict(venv='.venv_chronos', pesi='~/.cache/eweld_chronos/chronos-2'),
    'Chronos-Bolt': dict(venv='.venv_chronos', pesi='~/.cache/eweld_chronos/chronos-bolt-base'),
    'TimesFM 2.5': dict(venv='.venv_timesfm', pesi='~/.cache/eweld_timesfm/timesfm-2.5-200m-pytorch'),
    'TimesFM 3.0': dict(venv='.venv_timesfm', pesi='~/.cache/eweld_timesfm/timesfm-3.0-pytorch'),
}
# Timestral (Mistral, ottobre 2026): registrato solo se il suo venv esiste, così i giri senza di lui non cambiano.
if (RADICE/'.venv_timestral/bin/python').is_file():
    PREADDESTRATI['Timestral 209M'] = dict(venv='.venv_timestral', pesi='~/.cache/eweld_timestral/timestral-209m')
for _nome in PREADDESTRATI:
    PARAMETRI[_nome] = dict(contesto=2048)   # ore di storico in ingresso (circa 12 settimane)

BASELINE = ['Persistenza', 'Naive 24h', 'Naive 168h']
SENZA_ADDESTRAMENTO = BASELINE+list(PREADDESTRATI)
MODELLI = BASELINE+list(PARAMETRI)
VALUTAZIONI = ['diretta', 'breve', 'one_step', 'ricorsiva']


def crea_modello(nome, parametri=None, seed=0, dispositivo='auto', thread=4):
    if nome in BASELINE:
        return Baseline(nome)
    if nome not in PARAMETRI:
        raise ValueError(f'Modello sconosciuto: {nome}. Disponibili: {MODELLI}')
    p = {**PARAMETRI[nome], **(parametri or {})}
    if nome in PREADDESTRATI:
        return Preaddestrato(nome, p)
    if nome in ('XGBoost', 'LightGBM'):
        return Albero(nome, p, seed, thread)
    return Rete(nome, p, seed, dispositivo)


# --------------------------------------------------------------------------- grafico della loss
class GraficoLoss:
    """Curva di loss train/validazione aggiornata in place nel notebook."""

    def __init__(self, titolo):
        import matplotlib.pyplot as plt
        self.titolo = titolo
        self.fig, self.ax = plt.subplots(figsize=(7, 3))
        plt.close(self.fig)
        self.handle = None

    def aggiorna(self, storia):
        ax = self.ax
        ax.clear()
        x = np.arange(1, len(storia['train'])+1)
        ax.plot(x, storia['train'], label='train', lw=1.5)
        ax.plot(x, storia['validazione'], label='validazione', lw=1.5)
        if storia.get('migliore') is not None:
            ax.axvline(storia['migliore']+1, color='grey', ls='--', lw=1, label='migliore')
        ax.set_title(self.titolo, fontsize=10)
        ax.set_xlabel(storia['unita'])
        ax.set_ylabel('MSE (scala standardizzata)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        from IPython.display import display
        if self.handle is None:
            self.handle = display(self.fig, display_id=True)
        else:
            self.handle.update(self.fig)


def _nuova_storia(unita):
    return dict(train=[], validazione=[], migliore=None, unita=unita)


# --------------------------------------------------------------------------- valutazioni
def previsione_ricorsiva(prevedi_z, storico, calendario_esteso, horizon):
    """Autoregressiva: usa solo l'uscita +1h e la reinserisce come ultimo valore dello storico."""
    storico = storico.copy()
    uscite = []
    for k in range(horizon):
        prossima = prevedi_z(storico, calendario_esteso[:, k:k+horizon])[:, 0]
        uscite.append(prossima)
        nuova_ora = np.concatenate([prossima[:, None], calendario_esteso[:, k]], axis=1)
        storico = np.concatenate([storico[:, 1:], nuova_ora[:, None].astype(storico.dtype)], axis=1)
    return np.stack(uscite, axis=1)


def valutazioni(prevedi_z, ingressi, horizon, ore_breve):
    """prevedi_z(storico, futuro) -> (n, H) standardizzato. Restituisce le quattro modalità in z."""
    orario = prevedi_z(ingressi['storico_orario'], ingressi['futuro_orario'])
    return {
        'diretta': prevedi_z(ingressi['storico_test'], ingressi['futuro_test']),
        'breve': orario[:, :ore_breve],
        'one_step': orario[:, :1],
        'ricorsiva': previsione_ricorsiva(prevedi_z, ingressi['storico_test'],
                                          ingressi['calendario_esteso'], horizon),
    }


def _ingressi_test(dati):
    return dict(storico_test=dati.test.storico, futuro_test=dati.test.futuro,
                storico_orario=dati.test_orario.storico, futuro_orario=dati.test_orario.futuro,
                calendario_esteso=dati.calendario_esteso)


class _Modello:
    storia = None

    def prevedi_z(self, storico, futuro):
        raise NotImplementedError

    def prevedi_test(self, dati):
        z = valutazioni(self.prevedi_z, _ingressi_test(dati), dati.horizon, dati.ore_breve)
        return {k: dati.a_kwh(v) for k, v in z.items()}


# --------------------------------------------------------------------------- baseline
class Baseline(_Modello):
    """Riferimenti senza parametri, costruiti dallo storico reale all'origine."""

    def __init__(self, nome):
        self.nome = nome

    def addestra(self, dati, grafico=None):
        return self

    def prevedi_z(self, storico, futuro):
        passato, h = storico[:, :, 0], futuro.shape[1]
        if self.nome == 'Persistenza':
            return np.repeat(passato[:, -1:], h, axis=1)
        periodo = 24 if self.nome == 'Naive 24h' else 168
        if passato.shape[1] < periodo:
            raise ValueError(f'{self.nome} richiede lookback >= {periodo}.')
        # Valore di un periodo prima di ogni ora futura; se h > periodo si ripete il ciclo.
        return np.stack([passato[:, -periodo+(k % periodo)] for k in range(h)], axis=1)


# --------------------------------------------------------------------------- alberi
def caratteristiche_alberi(storico, futuro):
    """Stesso storico delle reti, in forma tabellare: lag, statistiche, calendario futuro."""
    v = storico[:, :, 0]
    statistiche = [stat(v[:, -min(w, v.shape[1]):], axis=1)
                   for w in (24, 168) for stat in (np.mean, np.std, np.min, np.max)]
    return np.concatenate([v, np.column_stack(statistiche), storico[:, -1, 1:],
                           futuro.reshape(len(futuro), -1)], axis=1).astype('float32')


class Albero(_Modello):
    def __init__(self, nome, parametri, seed, thread):
        self.nome, self.parametri, self.seed, self.thread = nome, parametri, seed, thread
        self._previsioni_z = None

    def addestra(self, dati, grafico=None):
        """Addestra e prevede il test nel processo separato (i pesi non tornano al kernel)."""
        richiesta = dict(nome=self.nome, parametri=self.parametri, seed=self.seed, thread=self.thread,
                         horizon=dati.horizon, ore_breve=dati.ore_breve)
        array = dict(x_train=caratteristiche_alberi(dati.train.storico, dati.train.futuro),
                     y_train=dati.train.target_z,
                     x_val=caratteristiche_alberi(dati.validazione.storico, dati.validazione.futuro),
                     y_val=dati.validazione.target_z, **_ingressi_test(dati))
        risultato = _esegui_processo_alberi(richiesta, array, self.thread)
        self.storia = dict(train=risultato['train'].tolist(), validazione=risultato['validazione'].tolist(),
                           migliore=int(risultato['migliore']), unita='iterazione (alberi)')
        self._previsioni_z = {k: risultato[k] for k in VALUTAZIONI}
        if grafico is not None:
            grafico.aggiorna(self.storia)
        return self

    def prevedi_test(self, dati):
        return {k: dati.a_kwh(v) for k, v in self._previsioni_z.items()}


def _esegui_processo_alberi(richiesta, array, thread):
    env = {k: str(thread) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'VECLIB_MAXIMUM_THREADS')}
    return _esegui_processo(richiesta, array, sys.executable, '--alberi', env)


def _esegui_processo(richiesta, array, python, modalita, env_extra=None):
    """Lancia questo file con l'interprete `python`; scambio dati tramite npz in una cartella temporanea."""
    with tempfile.TemporaryDirectory(prefix='c24_') as cartella:
        cartella = Path(cartella)
        (cartella/'richiesta.json').write_text(json.dumps(richiesta))
        np.savez(cartella/'input.npz', **array)
        env = {**os.environ, **(env_extra or {})}
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        processo = subprocess.run([str(python), str(Path(__file__).resolve()), modalita, str(cartella)],
                                  env=env, capture_output=True, text=True)
        if processo.returncode != 0:
            raise RuntimeError(f"{richiesta['nome']}: processo terminato con codice {processo.returncode}\n"
                               f'{processo.stderr[-3000:]}')
        with np.load(cartella/'output.npz') as out:
            return {k: out[k] for k in out.files}


def _lavoratore_alberi(cartella):
    """Eseguito solo nel processo separato: qui non si importa PyTorch."""
    cartella = Path(cartella)
    r = json.loads((cartella/'richiesta.json').read_text())
    with np.load(cartella/'input.npz') as d:
        xtr, ytr, xva, yva = (d[k] for k in ('x_train', 'y_train', 'x_val', 'y_val'))
        ingressi = {k: d[k] for k in d.files if not k.startswith(('x_', 'y_'))}
    p = dict(r['parametri'])
    if r['nome'] == 'XGBoost':
        import xgboost as xgb
        # Target multi-output con un albero per orizzonte a ogni iterazione: strategia diretta.
        m = xgb.XGBRegressor(**p, tree_method='hist', multi_strategy='one_output_per_tree',
                             eval_metric='rmse', n_jobs=r['thread'], random_state=r['seed'])
        m.fit(xtr, ytr, eval_set=[(xtr, ytr), (xva, yva)], verbose=False)
        curve = m.evals_result()
        train = np.square(curve['validation_0']['rmse'])
        val = np.square(curve['validation_1']['rmse'])
        migliore = int(m.best_iteration)

        def prevedi_z(storico, futuro):
            return m.predict(caratteristiche_alberi(storico, futuro), iteration_range=(0, migliore+1))
    else:
        import lightgbm as lgb
        # Un regressore per orizzonte; il numero di alberi si sceglie sulla curva media di validazione.
        modelli, train, val = [], 0, 0
        for h in range(ytr.shape[1]):
            m = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=r['thread'], random_state=r['seed'])
            m.fit(xtr, ytr[:, h], eval_X=(xtr, xva), eval_y=(ytr[:, h], yva[:, h]),
                  eval_names=['train', 'val'], eval_metric='l2')
            train = train+np.asarray(m.evals_result_['train']['l2'])/ytr.shape[1]
            val = val+np.asarray(m.evals_result_['val']['l2'])/ytr.shape[1]
            modelli.append(m)
        migliore = int(np.argmin(val))

        def prevedi_z(storico, futuro):
            x = caratteristiche_alberi(storico, futuro)
            return np.column_stack([m.predict(x, num_iteration=migliore+1) for m in modelli])
    previsioni = valutazioni(prevedi_z, ingressi, r['horizon'], r['ore_breve'])
    np.savez(cartella/'output.npz', train=train, validazione=val, migliore=migliore, **previsioni)


# --------------------------------------------------------------------------- reti ricorrenti
def _dispositivo(scelta):
    import torch
    if scelta == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    return scelta


def _rete_ricorrente(tipo, n_ingressi, n_calendario, horizon, hidden, dropout):
    import torch
    from torch import nn

    class ReteRicorrente(nn.Module):
        """Encoder LSTM/GRU sullo storico; testa condivisa su contesto, calendario futuro e orizzonte."""

        def __init__(self):
            super().__init__()
            ricorrente = nn.LSTM if tipo == 'LSTM' else nn.GRU
            self.encoder = ricorrente(n_ingressi, hidden, batch_first=True)
            self.testa = nn.Sequential(nn.Linear(hidden+n_calendario+1, hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, storico, futuro):
            _, stato = self.encoder(storico)
            nascosto = stato[0] if isinstance(stato, tuple) else stato
            contesto = nascosto[-1].unsqueeze(1).expand(-1, horizon, -1)
            passo = torch.linspace(1/horizon, 1, horizon, device=storico.device, dtype=storico.dtype)
            passo = passo.view(1, horizon, 1).expand(storico.shape[0], -1, -1)
            return self.testa(torch.cat([contesto, futuro, passo], dim=-1)).squeeze(-1)

    return ReteRicorrente()


class Rete(_Modello):
    def __init__(self, nome, parametri, seed, dispositivo):
        self.nome, self.parametri, self.seed = nome, parametri, seed
        self.dispositivo = dispositivo
        self.rete = None

    def addestra(self, dati, grafico=None):
        import torch
        p = self.parametri
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device(_dispositivo(self.dispositivo))
        self.dispositivo = str(dev)
        rete = _rete_ricorrente(self.nome, dati.train.storico.shape[2], dati.train.futuro.shape[2],
                                dati.horizon, p['hidden'], p['dropout']).to(dev)
        ottimizzatore = torch.optim.AdamW(rete.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])

        def tensori(f):
            return [torch.as_tensor(a, device=dev) for a in (f.storico, f.futuro, f.target_z)]

        xtr, ftr, ytr = tensori(dati.train)
        xva, fva, yva = tensori(dati.validazione)
        generatore = torch.Generator().manual_seed(self.seed)
        storia = _nuova_storia('epoca')
        migliore, stato_migliore, attesa = np.inf, None, 0
        for epoca in range(p['epoche']):
            rete.train()
            ordine = torch.randperm(len(ytr), generator=generatore).to(dev)
            somma = 0.0
            for i in range(0, len(ytr), p['batch']):
                b = ordine[i:i+p['batch']]
                loss = torch.mean((rete(xtr[b], ftr[b])-ytr[b])**2)
                ottimizzatore.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rete.parameters(), 1.0)
                ottimizzatore.step()
                somma += loss.item()*len(b)
            rete.eval()
            with torch.no_grad():
                loss_val = torch.mean((rete(xva, fva)-yva)**2).item()
            storia['train'].append(somma/len(ytr))
            storia['validazione'].append(loss_val)
            if loss_val < migliore:
                migliore, attesa, storia['migliore'] = loss_val, 0, epoca
                stato_migliore = {k: v.detach().clone() for k, v in rete.state_dict().items()}
            else:
                attesa += 1
            if grafico is not None:
                grafico.aggiorna(storia)
            if attesa >= p['pazienza']:
                break
        # Early stopping: si tengono i pesi dell'epoca con la loss di validazione minima.
        rete.load_state_dict(stato_migliore)
        self.rete, self.storia = rete, storia
        return self

    def prevedi_z(self, storico, futuro):
        import torch
        dev = next(self.rete.parameters()).device
        self.rete.eval()
        with torch.no_grad():
            return self.rete(torch.as_tensor(storico, device=dev),
                             torch.as_tensor(futuro, device=dev)).cpu().numpy()


# --------------------------------------------------------------------------- preaddestrati
class Preaddestrato(_Modello):
    """Foundation model zero-shot: riceve solo lo storico del consumo (niente calendario) e prevede.

    Le quattro valutazioni sono le stesse degli altri modelli; nelle origini orarie prevede
    +1…+ore_breve (la one-step-ahead ne usa la prima ora); nella ricorsiva il modello prevede
    +1h, la previsione viene aggiunta in coda al contesto e si ripete H volte.
    """

    def __init__(self, nome, parametri):
        self.nome, self.parametri = nome, parametri
        self._previsioni = None

    def addestra(self, dati, grafico=None):
        from dati import contesti
        c, h = self.parametri['contesto'], dati.horizon
        vuoto = lambda n, ore: np.zeros((n, ore, 0), dtype='float32')  # nessuna variabile di calendario
        test, orario = contesti(dati, dati.test.origini, c), contesti(dati, dati.test_orario.origini, c)
        ingressi = dict(storico_test=test[:, :, None], futuro_test=vuoto(len(test), h),
                        storico_orario=orario[:, :, None], futuro_orario=vuoto(len(orario), dati.ore_breve),
                        calendario_esteso=vuoto(len(test), 2*h-1))
        info = PREADDESTRATI[self.nome]
        python = RADICE/info['venv']/'bin/python'
        if not python.is_file():
            raise FileNotFoundError(f'{self.nome}: venv mancante {python}')
        richiesta = dict(nome=self.nome, pesi=os.path.expanduser(info['pesi']), horizon=h, contesto=c,
                         ore_breve=dati.ore_breve)
        # Il modello lavora direttamente in kWh: normalizza internamente ogni serie.
        self._previsioni = _esegui_processo(richiesta, ingressi, python, '--preaddestrato')
        return self

    def prevedi_test(self, dati):
        return {k: self._previsioni[k] for k in VALUTAZIONI}


def _carica_preaddestrato(nome, pesi, contesto):
    """Restituisce prevedi(serie: list[np.ndarray], horizon) -> (n, horizon), mediana/punto del modello."""
    if not os.path.isfile(os.path.join(pesi, 'model.safetensors')):
        raise FileNotFoundError(f'Pesi mancanti in {pesi}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    if nome.startswith('Chronos'):
        import torch
        from chronos import BaseChronosPipeline
        dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        pipeline = BaseChronosPipeline.from_pretrained(pesi, device_map=dev, dtype=torch.float32)

        def prevedi(serie, horizon, batch=64):
            uscite = []
            for i in range(0, len(serie), batch):
                q, _ = pipeline.predict_quantiles([torch.tensor(x) for x in serie[i:i+batch]],
                                                  prediction_length=horizon, quantile_levels=[.5])
                if isinstance(q, list):  # Chronos-2: un tensore (variate, h, q) per serie
                    q = torch.stack([x[0] for x in q])
                uscite.append(q[..., 0].float().cpu().numpy())
            return np.concatenate(uscite)
        return prevedi
    if nome.startswith('Timestral'):
        from timestral import TimestralForecaster
        modello = TimestralForecaster.from_pretrained(pesi)

        def prevedi(serie, horizon, batch=32):
            uscite = []
            for i in range(0, len(serie), batch):
                blocco = serie[i:i+batch]
                lunghezza = max(len(x) for x in blocco)
                contesto_b = np.full((len(blocco), lunghezza), np.nan, dtype='float32')
                for j, x in enumerate(blocco):   # serie più corte: NaN a sinistra (= valore mancante)
                    contesto_b[j, lunghezza-len(x):] = x
                m = modello.forecast(contesto_b, prediction_length=horizon).median
                m = m.detach().cpu().numpy() if hasattr(m, 'detach') else np.asarray(m)
                uscite.append(m.reshape(len(blocco), horizon))
            return np.concatenate(uscite)
        return prevedi
    if nome == 'TimesFM 3.0':
        from timesfm3.mlx import TimesFM3Forecaster
        modello = TimesFM3Forecaster.from_pretrained(pesi)

        def prevedi(serie, horizon, batch=64):
            return np.asarray([np.asarray(o.forecast) for i in range(0, len(serie), batch)
                               for o in modello.predict_batch(serie[i:i+batch], horizon=horizon)])
        return prevedi
    import torch
    import timesfm
    torch.set_num_threads(4)
    modello = timesfm.TimesFM_2p5_200M_torch.from_pretrained(pesi)
    modello.compile(timesfm.ForecastConfig(max_context=contesto, max_horizon=128, normalize_inputs=True,
                                           use_continuous_quantile_head=True, force_flip_invariance=True,
                                           infer_is_positive=True, fix_quantile_crossing=True,
                                           per_core_batch_size=64))

    def prevedi(serie, horizon):
        punto, _ = modello.forecast(horizon=horizon, inputs=list(serie))
        return np.asarray(punto)[:, :horizon]
    return prevedi


def _lavoratore_preaddestrato(cartella):
    """Eseguito nel venv del modello: carica i pesi una volta e calcola le quattro valutazioni."""
    cartella = Path(cartella)
    r = json.loads((cartella/'richiesta.json').read_text())
    with np.load(cartella/'input.npz') as d:
        ingressi = {k: d[k] for k in d.files}
    prevedi = _carica_preaddestrato(r['nome'], r['pesi'], r['contesto'])

    def prevedi_z(storico, futuro):
        return prevedi([x.astype('float32') for x in storico[:, :, 0]], futuro.shape[1])
    np.savez(cartella/'output.npz', **valutazioni(prevedi_z, ingressi, r['horizon'], r['ore_breve']))


# --------------------------------------------------------------------------- riepilogo
def riepilogo_addestramento(modelli, secondi):
    """Epoca/iterazione scelta e loss corrispondenti: se train scende e validazione sale, overfitting."""
    import pandas as pd
    righe = {}
    for nome, m in modelli.items():
        s = m.storia
        riga = {'unità': '—', 'eseguite': 0, 'migliore': np.nan,
                'loss train': np.nan, 'loss validazione': np.nan}
        if s is not None:
            b = s['migliore']
            riga = {'unità': s['unita'].split()[0], 'eseguite': len(s['train']), 'migliore': b+1,
                    'loss train': s['train'][b], 'loss validazione': s['validazione'][b]}
        righe[nome] = {**riga, 'secondi': secondi.get(nome, 0.0)}
    return pd.DataFrame(righe).T


if __name__ == '__main__' and len(sys.argv) == 3:
    {'--alberi': _lavoratore_alberi, '--preaddestrato': _lavoratore_preaddestrato}[sys.argv[1]](sys.argv[2])

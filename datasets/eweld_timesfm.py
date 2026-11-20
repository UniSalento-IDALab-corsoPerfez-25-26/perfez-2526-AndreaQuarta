"""Worker TimesFM zero-shot: va eseguito con il Python di `.venv_timesfm`, non con quello del notebook.

Uso: python eweld_timesfm.py MODELLO input.npz output.npz
  MODELLO: 'timesfm3' (TimesFM 3.0, backend MLX sulla GPU Apple) oppure 'timesfm25' (TimesFM 2.5, PyTorch su CPU).
input.npz: `values` (n, contesto) con NaN iniziali per i contesti più corti, `horizon` (scalare).
output.npz: `point` (n, horizon) mediana, `quantiles` (n, horizon, 9) decili 0.1..0.9.
Solo numpy: nessuna dipendenza dai moduli del progetto.
"""
import os
import sys
import time
import numpy as np

# Pesi scaricati una volta in locale (vedi `download_weights`): il download via huggingface_hub/hf-xet
# si bloccava su questa rete. Con i pesi in locale non serve alcun accesso a internet.
WEIGHTS_DIR = os.path.expanduser(os.environ.get('EWELD_TIMESFM_WEIGHTS', '~/.cache/eweld_timesfm'))
REPOS = {'timesfm3': 'timesfm-3.0-pytorch', 'timesfm25': 'timesfm-2.5-200m-pytorch'}


def checkpoint(name):
    path = os.path.join(WEIGHTS_DIR, REPOS[name])
    if not os.path.isfile(os.path.join(path, 'model.safetensors')):
        raise FileNotFoundError(f'Pesi mancanti in {path}: eseguire download_weights() dal notebook.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    return path


def download_weights(names=('timesfm3', 'timesfm25')):
    """Scarica config e pesi da Hugging Face via HTTP (curl), riprendendo i download interrotti."""
    import subprocess
    for name in names:
        path = os.path.join(WEIGHTS_DIR, REPOS[name])
        os.makedirs(path, exist_ok=True)
        for f in ('config.json', 'model.safetensors'):
            if not os.path.isfile(os.path.join(path, f)) or f == 'model.safetensors':
                subprocess.run(['curl', '-sS', '-L', '--retry', '3', '-C', '-', '-o', os.path.join(path, f),
                                f'https://huggingface.co/google/{REPOS[name]}/resolve/main/{f}'], check=True)
        print(name, 'pesi in', path)


def fill_gaps(row):
    """Interpolazione lineare dei NaN (agli estremi si ripete il valore più vicino); tutto NaN → zeri."""
    bad = ~np.isfinite(row)
    if bad.all():
        return np.zeros(len(row), np.float32)
    if bad.any():
        idx = np.arange(len(row))
        row = row.copy()
        row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return row.astype(np.float32)


def context_starts(values):
    """Indice del primo valore osservato di ogni contesto (i NaN iniziali vengono tolti)."""
    return [int(np.argmax(np.isfinite(row))) if np.isfinite(row).any() else len(row)-1 for row in values]


def contexts(values):
    """Toglie i NaN iniziali e interpola linearmente eventuali buchi interni (solo valori passati)."""
    return [fill_gaps(row[start:]) for row, start in zip(values, context_starts(values))]


def run_timesfm3(series, horizon, batch):
    from timesfm3.mlx import TimesFM3Forecaster
    model = TimesFM3Forecaster.from_pretrained(checkpoint('timesfm3'))
    point, quant = [], []
    for start in range(0, len(series), batch):
        for out in model.predict_batch(series[start:start+batch], horizon=horizon, return_quantiles=True):
            point.append(np.asarray(out.forecast))
            quant.append(np.asarray(out.quantiles))
        print(f'  timesfm3: {min(start+batch, len(series))}/{len(series)}', flush=True)
    return np.asarray(point), np.asarray(quant)


def run_timesfm25(series, horizon, batch, max_context):
    import torch
    import timesfm
    torch.set_num_threads(4)  # lascia spazio agli altri addestramenti sulla CPU
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint('timesfm25'))
    model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=max(128, horizon),
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True,
                                         fix_quantile_crossing=True, per_core_batch_size=batch))
    # forecast() allunga la lista ricevuta per completare il batch: gli si passa una copia.
    point, quant = model.forecast(horizon=horizon, inputs=list(series))
    # Quantili 2.5: indice 0 = media, 1..9 = decili.
    return np.asarray(point)[:, :horizon], np.asarray(quant)[:, :horizon, 1:10]


def run_from_notebook(name, panel, origins_by_user, cfg, python, context_hours=2048, workdir=None):
    """Lato notebook (venv del progetto): prepara i contesti, lancia il worker con il Python di
    `.venv_timesfm` e restituisce righe compatibili con eweld_robusta.metrics (più i decili in `quantiles`).
    Il contesto contiene solo ore precedenti all'origine; i buchi vengono interpolati nel worker."""
    label = {'timesfm3': 'TimesFM3_zeroshot', 'timesfm25': 'TimesFM25_zeroshot'}[name]
    return run_worker(__file__, name, label, panel, origins_by_user, cfg, python, context_hours, workdir)


def run_worker(script, name, label, panel, origins_by_user, cfg, python, context_hours=2048, workdir=None,
               covariates=None):
    """Comune a TimesFM e Chronos: `script` è il worker da lanciare con l'interprete `python`.
    `covariates` (facoltativo): {utente: DataFrame orario di variabili esterne}; per ogni origine si inviano le
    `context_hours` ore passate più le `horizon` ore future, ed è il worker a decidere se usare il futuro."""
    import subprocess
    import tempfile
    from pathlib import Path
    import pandas as pd
    h = cfg['horizon']
    keys, values, actual, cov = [], [], [], []
    for user in sorted(origins_by_user):
        s = panel[user]
        for origin in origins_by_user[user]:
            past = pd.date_range(origin-pd.Timedelta(context_hours, unit='h'), periods=context_hours, freq='h')
            keys.append((user, origin))
            values.append(s.reindex(past).to_numpy(float))
            actual.append(s.reindex(pd.date_range(origin, periods=h, freq='h')).to_numpy(float))
            if covariates is not None:
                cov.append(covariates[user].reindex(pd.date_range(past[0], periods=context_hours+h, freq='h'))
                           .to_numpy(np.float32))
    extra = {} if covariates is None else dict(covariates=np.asarray(cov),
                                               cov_names=np.asarray(next(iter(covariates.values())).columns, str))
    with tempfile.TemporaryDirectory(prefix='eweld_timesfm_', dir=workdir) as temp:
        source, target = Path(temp)/'input.npz', Path(temp)/'output.npz'
        np.savez(source, values=np.asarray(values), horizon=h, **extra)
        process = subprocess.run([str(python), str(Path(script).resolve()), name, str(source), str(target)],
                                 capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f'{name} fallito: {process.stderr[-4000:]}')
        print(process.stdout.strip().splitlines()[-1])
        with np.load(target) as out:
            point, quantiles = out['point'], out['quantiles']
    return [dict(model=label, seed=-1, user=u, origin=o, pred=p, actual=a, quantiles=q)
            for (u, o), p, a, q in zip(keys, point, actual, quantiles)]

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('timesfm3', 'timesfm25'):
        raise SystemExit(__doc__)
    name, source, target = sys.argv[1:]
    with np.load(source) as data:
        values, horizon = data['values'].astype(float), int(data['horizon'])
    series = contexts(values)
    t = time.perf_counter()
    if name == 'timesfm3':
        point, quant = run_timesfm3(series, horizon, batch=32)
    else:
        point, quant = run_timesfm25(series, horizon, batch=32, max_context=values.shape[1])
    if point.shape != (len(series), horizon) or quant.shape != (len(series), horizon, 9):
        raise ValueError(f'Forme inattese: {point.shape} {quant.shape}')
    if not np.isfinite(point).all():
        raise ValueError('Previsioni non finite')
    np.savez(target, point=point, quantiles=quant)
    print(f'{name}: {len(series)} previsioni in {time.perf_counter()-t:.0f}s', flush=True)

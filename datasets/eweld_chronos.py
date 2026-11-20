"""Worker Chronos zero-shot (Amazon): va eseguito con il Python di `.venv_chronos`.

Uso: python eweld_chronos.py MODELLO input.npz output.npz
  MODELLO: 'chronos2' (Chronos-2) oppure 'chronosbolt' (Chronos-Bolt base). Licenza Apache 2.0 per entrambi.
  Varianti di Chronos-2 con variabili esterne (input.npz deve contenere `covariates` e `cov_names`):
    'chronos2_meteo': solo il meteo passato (covariate note fino all'origine);
    'chronos2_meteo_oracolo': anche il meteo osservato nelle 48 ore future, cioè una previsione meteo perfetta.
Formato di input/output identico a eweld_timesfm.py (decili 0.1..0.9, mediana come previsione puntuale).
I pesi stanno in ~/.cache/eweld_chronos (scaricati via HTTP, verificati con SHA-256).
"""
import os
import sys
import time
import numpy as np

from eweld_timesfm import contexts, context_starts, fill_gaps  # stessa preparazione del contesto di TimesFM

WEIGHTS_DIR = os.path.expanduser(os.environ.get('EWELD_CHRONOS_WEIGHTS', '~/.cache/eweld_chronos'))
REPOS = {'chronos2': 'chronos-2', 'chronosbolt': 'chronos-bolt-base',
         'chronos2_meteo': 'chronos-2', 'chronos2_meteo_oracolo': 'chronos-2'}
LABELS = {'chronos2': 'Chronos2_zeroshot', 'chronosbolt': 'ChronosBolt_zeroshot',
          'chronos2_meteo': 'Chronos2_meteo', 'chronos2_meteo_oracolo': 'Chronos2_meteo_oracolo'}


def download_weights(names=('chronos2', 'chronosbolt')):
    """Scarica config e pesi da Hugging Face via HTTP (curl), riprendendo i download interrotti."""
    import subprocess
    for name in names:
        path = os.path.join(WEIGHTS_DIR, REPOS[name])
        os.makedirs(path, exist_ok=True)
        for f in ('config.json', 'model.safetensors'):
            subprocess.run(['curl', '-sS', '-L', '-f', '--retry', '3', '-C', '-', '-o', os.path.join(path, f),
                            f'https://huggingface.co/amazon/{REPOS[name]}/resolve/main/{f}'], check=True)
        print(name, 'pesi in', path)


def run_from_notebook(name, panel, origins_by_user, cfg, python, context_hours=2048, workdir=None, covariates=None):
    from eweld_timesfm import run_worker
    if name.startswith('chronos2_meteo') and covariates is None:
        raise ValueError(f'{name} richiede `covariates`.')
    return run_worker(__file__, name, LABELS[name], panel, origins_by_user, cfg, python, context_hours, workdir,
                      covariates=covariates)


def with_covariates(series, starts, covariates, names, horizon, future):
    """Input a dizionari di Chronos-2: il meteo è allineato al contesto (stessi NaN iniziali tolti) e, solo
    con `future=True`, esteso alle `horizon` ore successive all'origine."""
    inputs = []
    for target, start, cov in zip(series, starts, covariates):
        cov = np.column_stack([fill_gaps(c) for c in cov[start:].T])  # contesto + orizzonte
        past, ahead = cov[:-horizon], cov[-horizon:]
        assert len(past) == len(target)
        item = dict(target=target, past_covariates={n: past[:, i] for i, n in enumerate(names)})
        if future:
            item['future_covariates'] = {n: ahead[:, i] for i, n in enumerate(names)}
        inputs.append(item)
    return inputs


def predict(name, series, horizon, batch=64):
    import torch
    from chronos import BaseChronosPipeline
    path = os.path.join(WEIGHTS_DIR, REPOS[name])
    if not os.path.isfile(os.path.join(path, 'model.safetensors')):
        raise FileNotFoundError(f'Pesi mancanti in {path}: eseguire download_weights() dal notebook.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pipeline = BaseChronosPipeline.from_pretrained(path, device_map=device, dtype=torch.float32)
    levels = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    quant = []
    for start in range(0, len(series), batch):
        chunk = series[start:start+batch]
        if not isinstance(chunk[0], dict):
            chunk = [torch.tensor(x) for x in chunk]
        q, _ = pipeline.predict_quantiles(chunk, prediction_length=horizon, quantile_levels=levels)
        if isinstance(q, list):  # Chronos-2: una tensor (variate, h, q) per serie
            q = torch.stack([x[0] for x in q])
        quant.append(q.float().cpu().numpy())
    quant = np.concatenate(quant)
    print(f'  {name} su {device}', flush=True)
    return quant[:, :, 4], quant


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in REPOS:
        raise SystemExit(__doc__)
    name, source, target = sys.argv[1:]
    with np.load(source) as data:
        values, horizon = data['values'].astype(float), int(data['horizon'])
        covariates = data['covariates'].astype(float) if 'covariates' in data.files else None
        names = [str(n) for n in data['cov_names']] if 'cov_names' in data.files else []
    series = contexts(values)
    if name.startswith('chronos2_meteo'):
        if covariates is None:
            raise SystemExit(f'{name}: covariate mancanti in {source}')
        series = with_covariates(series, context_starts(values), covariates, names, horizon,
                                 future=name.endswith('oracolo'))
    t = time.perf_counter()
    point, quant = predict(name, series, horizon)
    if point.shape != (len(series), horizon) or quant.shape != (len(series), horizon, 9):
        raise ValueError(f'Forme inattese: {point.shape} {quant.shape}')
    if not np.isfinite(point).all():
        raise ValueError('Previsioni non finite')
    np.savez(target, point=point, quantiles=quant)
    print(f'{name}: {len(series)} previsioni in {time.perf_counter()-t:.0f}s', flush=True)

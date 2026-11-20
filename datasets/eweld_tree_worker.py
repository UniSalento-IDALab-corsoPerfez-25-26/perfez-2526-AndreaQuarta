"""Training isolato degli alberi EWELD. Avviato dal notebook con sys.executable."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('arrays')
    parser.add_argument('result')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    # Importare solo il backend richiesto, mai torch o l'altro backend ad alberi.
    import numpy as np
    import pandas as pd
    from sklearn.multioutput import MultiOutputRegressor
    name = request['name']
    if name == 'XGBoost':
        from xgboost import XGBRegressor
        constructor = XGBRegressor
        fixed = dict(objective='reg:squarederror', tree_method='hist', subsample=.9, colsample_bytree=.9)
    elif name == 'LightGBM':
        from lightgbm import LGBMRegressor
        constructor = LGBMRegressor
        fixed = dict(objective='regression', min_child_samples=15, verbosity=-1,
                     deterministic=True, force_col_wise=True)
    else:
        raise ValueError(f'Modello non supportato: {name}')
    fixed.update(n_jobs=request['threads'], random_state=request['seed'])
    with np.load(args.arrays, allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
    cols = request['columns']
    frame = lambda key: pd.DataFrame(np.ascontiguousarray(a[key]), columns=cols)
    xfit, xval = frame('xfit'), frame('xval')
    rows, candidates = [], []
    start_all = perf_counter()
    for cid, cfg in enumerate(request['grid']):
        start = perf_counter()
        model = None
        try:
            model = MultiOutputRegressor(constructor(**fixed, **cfg), n_jobs=1)
            model.fit(xfit, np.asfortranarray(a['yfit']))
            pred = model.predict(xval)*request['selection_scale']+request['selection_mean']
            if pred.shape != a['validation_targets'].shape or not np.isfinite(pred).all():
                raise ValueError('Previsione di validazione non valida.')
            score = float(np.sqrt(np.mean((pred-a['validation_targets'])**2, axis=1)).mean())
            candidates.append((score, cid, cfg))
            state = 'OK'
        except Exception as exc:
            score, state = None, f'{type(exc).__name__}: {exc}'
        rows.append(dict(modello=name, candidato=cid, RMSE_val=score, epoca=None,
                         parametri=json.dumps(cfg), secondi=perf_counter()-start, stato=state))
        print(f'{name} candidato {cid+1}: RMSE val={score} | {state}', flush=True)
        del model
    if not candidates:
        Path(args.result).write_text(json.dumps({'search': rows, 'error': 'Nessuna configurazione valida'}))
        return
    score, cid, cfg = min(candidates, key=lambda row: (row[0], row[1]))
    model = MultiOutputRegressor(constructor(**fixed, **cfg), n_jobs=1)
    model.fit(frame('xfinal'), np.asfortranarray(a['yfinal']))
    pred = (model.predict(frame('xpredict'))*request['final_scale']+request['final_mean'])[0]
    if not np.isfinite(pred).all():
        raise ValueError('Refit non finito.')
    result = dict(search=rows, prediction=pred.tolist(), selection=dict(parametri=cfg, RMSE_val=score,
                  epoche=None, device='cpu / processo isolato', secondi=perf_counter()-start_all))
    Path(args.result).write_text(json.dumps(result))


if __name__ == '__main__':
    main()

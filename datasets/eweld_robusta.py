"""Valutazione robusta C24: rolling-origin su un anno, tutti gli utenti ammessi, test statistici.

Riusa dati, calendario e worker XGBoost/GRU/TFT di `eweld_multiutente.py` (non modificato).
Aggiunge: campioni tolleranti ai buchi, LightGBM, SARIMAX, refit periodico, Diebold-Mariano,
Friedman/Nemenyi, Wilcoxon-Holm, bootstrap a blocchi e analisi della selezione dell'utente.
Nessun training all'import.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import eweld_multiutente as bench

H = pd.Timedelta(1, unit='h')


# --------------------------------------------------------------------------- calendario del test
def test_origins(cfg):
    """Origini non sovrapposte: ogni blocco di `horizon` ore cade interamente nel periodo di test."""
    start = pd.Timestamp(cfg['test_start'])
    last = pd.Timestamp(cfg['test_end'])+pd.Timedelta(1, unit='D')-cfg['horizon']*H
    origins = pd.date_range(start, last, freq=f"{cfg['origin_step']}h")
    if cfg.get('max_origins'):
        origins = origins[:cfg['max_origins']]
    return origins


def refit_dates(cfg, origins):
    dates = pd.date_range(pd.Timestamp(cfg['test_start']), origins[-1], freq=f"{cfg['refit_days']}D")
    return dates[:cfg['max_refits']] if cfg.get('max_refits') else dates


def assign_refit(origins, refits):
    """Ogni origine usa l'ultimo modello riaddestrato prima di essa (solo dati passati)."""
    pos = np.searchsorted(refits.values, origins.values, side='right')-1
    if (pos < 0).any():
        raise ValueError('Origine precedente al primo refit.')
    return pd.Series(refits[pos], index=origins)


def window_ok(series, origin, cfg):
    idx = pd.date_range(origin-cfg['lookback']*H, periods=cfg['lookback']+cfg['horizon'], freq='h')
    return not series.reindex(idx).isna().any()


def eligibility(panel, cfg):
    """Criterio di sola disponibilità dei dati, fissato prima di osservare gli errori."""
    t0 = pd.Timestamp(cfg['test_start'])
    train_idx = pd.date_range(t0-pd.Timedelta(cfg['train_days'], unit='D'), t0-H, freq='h')
    origins = test_origins(cfg)
    rows = []
    for user, s in panel.items():
        train = s.reindex(train_idx)
        train_cov = float(1-train.isna().mean())
        # Attività: quota di ore a zero nel solo training (contatori spenti o impianti fermi tutto l'anno).
        zero_share = float((train.dropna() <= 0).mean()) if train.notna().any() else 1.
        valid = int(sum(window_ok(s, o, cfg) for o in origins))
        rows.append(dict(user=user, train_coverage=train_cov, train_zero_share=zero_share, valid_origins=valid,
                         origin_coverage=valid/len(origins),
                         eligible=(train_cov >= cfg['min_train_coverage'] and
                                   zero_share <= cfg['max_train_zero_share'] and
                                   valid/len(origins) >= cfg['min_origin_coverage'])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- campioni
def scaling(history):
    """Come bench.scaling, ma tollera ore mancanti nel training."""
    a = history.to_numpy(float)
    mean, std = np.nanmean(a), np.nanstd(a)
    std = max(float(std), 1e-6)
    raw = float(np.nanmean(np.abs(a[24:]-a[:-24])))
    floor = max(1e-6, .001*float(np.nanmean(np.abs(a))))
    return dict(mean=float(mean), std=std, raw_scale=raw, scale=max(raw, floor), scale_floored=raw < floor)


def _features(series, origin, params, cfg):
    past_idx = pd.date_range(origin-cfg['lookback']*H, periods=cfg['lookback'], freq='h')
    future_idx = pd.date_range(origin, periods=cfg['horizon'], freq='h')
    past = series.reindex(past_idx).to_numpy(float)
    x = np.column_stack([((past-params['mean'])/params['std'])[:, None], bench.calendar(past_idx)])
    target = series.reindex(future_idx).to_numpy(float)
    return x, bench.calendar(future_idx), target


def _pack(rows, all_users, city):
    if not rows:
        raise ValueError('Nessun esempio disponibile.')
    users = [r['user'] for r in rows]
    static = [np.r_[np.eye(len(all_users), dtype='float32')[all_users.index(u)],
                    np.eye(3, dtype='float32')[city[u]-1]] for u in users]
    out = dict(x=np.asarray([r['x'] for r in rows], dtype='float32'),
               f=np.asarray([r['f'] for r in rows], dtype='float32'),
               y=np.asarray([r['y'] for r in rows], dtype='float32'),
               static=np.asarray(static, dtype='float32'),
               mean=np.asarray([r['mean'] for r in rows]), std=np.asarray([r['std'] for r in rows]),
               scale=np.asarray([r['scale'] for r in rows]),
               user=np.asarray(users), origin=np.asarray([str(r['origin']) for r in rows]))
    counts = pd.Series(users).value_counts()
    weight = np.array([1/counts[u] for u in users])
    out['weight'] = (weight/weight.mean()).astype('float32')
    return out


def training_samples(panel, users, all_users, city, start, end, cfg):
    """Finestre complete in [start, end): stesso numero di esempi per utente (richiesto dai worker)."""
    per_user, params = {}, {}
    for user in sorted(users):
        hist = panel[user].loc[start:end-H]
        params[user] = scaling(hist)
        full = panel[user].reindex(pd.date_range(start, end-H, freq='h'))
        ok = ~full.isna().to_numpy()
        csum = np.r_[0, np.cumsum(ok)]
        length = cfg['lookback']+cfg['horizon']
        positions = np.arange(cfg['lookback'], len(full)-cfg['horizon']+1, cfg['stride'])
        good = positions[(csum[positions+cfg['horizon']]-csum[positions-cfg['lookback']]) == length]
        per_user[user] = full.index[good]
    n = min(cfg['max_samples_per_user'], min(len(v) for v in per_user.values()))
    if n < cfg['min_samples_per_user']:
        raise ValueError(f'Troppi pochi esempi di training ({n}) tra {start} e {end}.')
    rows = []
    for user in sorted(users):
        chosen = per_user[user][np.linspace(0, len(per_user[user])-1, n, dtype=int)]
        p = params[user]
        for origin in chosen:
            x, f, target = _features(panel[user], origin, p, cfg)
            rows.append(dict(user=user, origin=origin, x=x, f=f, y=(target-p['mean'])/p['std'], **p))
    return _pack(rows, all_users, city), params


def prediction_samples(panel, origins_by_user, all_users, city, params, cfg, with_target):
    """Ingressi solo passati. Senza `with_target` il target inviato al worker è zero (mai il reale)."""
    rows, actual = [], []
    for user in sorted(origins_by_user):
        p = params[user]
        for origin in origins_by_user[user]:
            x, f, target = _features(panel[user], origin, p, cfg)
            if np.isnan(x).any():
                raise ValueError(f'Contesto incompleto {user} {origin}')
            y = (target-p['mean'])/p['std'] if with_target else np.zeros(cfg['horizon'])
            rows.append(dict(user=user, origin=origin, x=x, f=f, y=y, **p))
            actual.append(target)
    return _pack(rows, all_users, city), np.asarray(actual)


# --------------------------------------------------------------------------- worker in processi isolati
def model_job(model, params, fit, predict, cfg, seed, tune):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='eweld_robusta_') as temp:
        temp = Path(temp)
        for label, data in [('fit', fit), ('predict', predict)]:
            np.savez(temp/f'{label}.npz', **data)
        request = dict(model=model, model_config=params, seed=int(seed), tune=tune, threads=cfg['threads'],
                       epochs=cfg['max_epochs'], patience=cfg['patience'], batch_size=cfg['batch_size'])
        (temp/'request.json').write_text(json.dumps(request))
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[key] = str(cfg['threads'])
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent)+os.pathsep+env.get('PYTHONPATH', '')
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(temp)],
                                 env=env, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f'{model} fallito: {process.stderr[-4000:]} {process.stdout[-2000:]}')
        result = json.loads((temp/'result.json').read_text())
    result['seconds'] = time.perf_counter()-start
    return result


def stop_features(data, low_fraction=.05):
    """Indicatori di fermo impianto dal solo contesto passato. Un'ora è 'bassa' se il consumo non supera
    il 5% della media di training dell'utente (stessa soglia dei regimi di bench.evaluate, qui sulla media)."""
    raw = data['x'][:, :, 0]*data['std'][:, None]+data['mean'][:, None]
    low = raw <= low_fraction*np.maximum(data['mean'][:, None], 1e-6)
    length = low.shape[1]

    def hours_since(mask):
        rev = mask[:, ::-1]
        return np.where(rev.any(axis=1), rev.argmax(axis=1), length).astype(float)
    return np.column_stack([low[:, -24:].mean(axis=1), low.mean(axis=1), low[:, -1],
                            hours_since(low), hours_since(~low),
                            (low[:, 1:] != low[:, :-1]).sum(axis=1)]).astype('float32')


def _worker(folder):
    folder = Path(folder)
    request = json.loads((folder/'request.json').read_text())
    if request['model'] != 'LightGBM':
        return bench._worker(folder)
    with np.load(folder/'fit.npz') as file:
        fit = {k: file[k] for k in file.files}
    with np.load(folder/'predict.npz') as file:
        val = {k: file[k] for k in file.files}
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    def features(data):  # stesse feature dell'XGBoost di bench
        x = data['x']
        parts = [x[:, :, 0], x[:, -1, 1:], data['f'].reshape(len(x), -1), data['static']]
        for width in (24, 168):
            for fn in (np.mean, np.std, np.min, np.max):
                parts.append(fn(x[:, -width:, :], axis=1))
        return np.concatenate(parts, axis=1)
    options = request['model_config']
    args = {k: v for k, v in options.items() if k not in ('epochs', 'fermi', 'residuo')}
    build = (lambda d: np.concatenate([features(d), stop_features(d)], axis=1)) if options.get('fermi') else features
    h = fit['y'].shape[1]
    # Residuo: il modello impara la correzione al naive giornaliero (stessa normalizzazione del target).
    naive = (lambda d: np.tile(d['x'][:, -24:, 0], (1, int(np.ceil(h/24))))[:, :h]) if options.get('residuo') \
        else (lambda d: 0.)
    model = MultiOutputRegressor(LGBMRegressor(**args, random_state=request['seed'], n_jobs=request['threads'],
                                               verbose=-1), n_jobs=1)
    model.fit(build(fit), fit['y']-naive(fit), sample_weight=fit['weight'])
    prediction = model.predict(build(val))+naive(val)
    if prediction.shape != val['y'].shape or not np.isfinite(prediction).all():
        raise ValueError('Output non valido')
    (folder/'result.json').write_text(json.dumps(dict(prediction=prediction.tolist(), epochs=0, history=[])))


# --------------------------------------------------------------------------- baseline
def baseline_forecasts(panel, origins_by_user, cfg):
    h, rows = cfg['horizon'], []
    for user, origins in origins_by_user.items():
        s = panel[user]
        for origin in origins:
            past = s.reindex(pd.date_range(origin-672*H, periods=672, freq='h')).to_numpy(float)
            actual = s.reindex(pd.date_range(origin, periods=h, freq='h')).to_numpy(float)
            n168 = np.resize(past[-168:], h)
            weeks = past.reshape(4, 168)[:, :h]
            # Media delle stesse ore delle ultime 4 settimane; se mancano tutte, naive settimanale.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mean4 = np.nanmean(weeks, axis=0)
            mean4 = np.where(np.isfinite(mean4), mean4, n168)
            for name, pred in [('Naive24', np.resize(past[-24:], h)), ('Naive168', n168),
                               ('MediaStagionale4s', mean4)]:
                rows.append(dict(model=name, seed=-1, user=user, origin=origin, pred=pred, actual=actual))
    return rows


# --------------------------------------------------------------------------- SARIMAX
def sarimax_exog(index):
    """Fourier settimanali (k=1..3) e indicatore del Capodanno lunare (da bench.calendar)."""
    week = (index.dayofweek.to_numpy()*24+index.hour.to_numpy())/168
    cols = [f(2*np.pi*k*week) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    return pd.DataFrame(np.column_stack(cols+[bench.calendar(index)[:, 6]]), index=index,
                        columns=[f'{f}{k}' for k in (1, 2, 3) for f in ('s', 'c')]+['festa'])


def sarimax_user(job):
    """Un utente: fit a ogni refit su `sarimax_train_days`, poi filtro senza refit a ogni origine."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')
    series, user, plan, cfg = job['series'], job['user'], job['plan'], job['cfg']
    days, h, rows = cfg['sarimax_train_days'], cfg['horizon'], []
    for refit, origins in plan:
        fit_idx = pd.date_range(refit-pd.Timedelta(days, unit='D'), refit-H, freq='h')
        exog = sarimax_exog(fit_idx)
        keep = [c for c in exog if exog[c].std() > 0]  # colonne costanti (festa assente) escluse
        result, error = None, None
        try:
            model = SARIMAX(series.reindex(fit_idx), exog=exog[keep], order=tuple(cfg['sarimax_order']),
                            seasonal_order=tuple(cfg['sarimax_seasonal']), enforce_stationarity=False,
                            enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=cfg['sarimax_maxiter'])
        except Exception as exc:  # registrato e sostituito con naive168, mai scartato in silenzio
            error = repr(exc)[:200]
        for origin in origins:
            idx = pd.date_range(origin-pd.Timedelta(days, unit='D'), origin-H, freq='h')
            fut = pd.date_range(origin, periods=h, freq='h')
            actual = series.reindex(fut).to_numpy(float)
            fallback = np.resize(series.reindex(pd.date_range(origin-168*H, periods=168, freq='h')).to_numpy(float), h)
            pred, status = fallback, 'fallback_fit' if error else 'ok'
            if result is not None:
                try:
                    applied = result.apply(series.reindex(idx), exog=sarimax_exog(idx)[keep], refit=False)
                    pred = applied.forecast(h, exog=sarimax_exog(fut)[keep]).to_numpy(float)
                    if not np.isfinite(pred).all() or np.abs(pred).max() > 1e3*max(np.nanmax(np.abs(series.reindex(idx))), 1):
                        pred, status = fallback, 'fallback_esplosa'
                except Exception:
                    pred, status = fallback, 'fallback_filtro'
            rows.append(dict(model='SARIMAX', seed=-1, user=user, origin=origin, pred=pred, actual=actual,
                             status=status))
    return rows


def run_sarimax(panel, origins_by_user, refit_of, cfg, workers):
    jobs = []
    for user, origins in origins_by_user.items():
        plan = [(r, [o for o in origins if refit_of[o] == r]) for r in sorted(set(refit_of[o] for o in origins))]
        start = min(origins)-pd.Timedelta(cfg['sarimax_train_days']+cfg['refit_days']+2, unit='D')
        jobs.append(dict(series=panel[user].loc[start:], user=user, plan=plan, cfg=cfg))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sarimax_user, j): j['user'] for j in jobs}
        for done in as_completed(futures):
            rows.extend(done.result())
            print(f'  SARIMAX {futures[done]} completato', flush=True)
    return rows


# --------------------------------------------------------------------------- modelli appresi
def _val_score(pred, data):
    losses = np.mean(np.abs(pred-data['y'])*data['std'][:, None], axis=1)/data['scale']
    per_user = pd.DataFrame({'user': data['user'], 'loss': losses}).groupby('user').loss.mean()
    return float(per_user.mean()), per_user.to_dict()


def tune_spec(spec, panel, users, city, cfg):
    """Tuning unico prima del test: fit [t0-365g, t0-14g), validazione interna sugli ultimi 14 giorni."""
    t0 = pd.Timestamp(cfg['test_start'])
    start = t0-pd.Timedelta(cfg['train_days'], unit='D')
    split = t0-cfg['val_blocks']*cfg['horizon']*H
    inner = pd.date_range(split, periods=cfg['val_blocks'], freq=f"{cfg['horizon']}h")
    groups = {u: [u] for u in users} if spec['strategy'] == 'locale' else {'all': list(users)}
    out = {}

    def one(group, members):
        fit, params = training_samples(panel, members, users, city, start, split, cfg)
        ok = {u: [o for o in inner if window_ok(panel[u], o, cfg)] for u in members}
        val, _ = prediction_samples(panel, {u: v for u, v in ok.items() if v}, users, city, params, cfg, True)
        best, trials = None, []
        for cid, candidate in enumerate(spec['grid']):
            answer = model_job(spec['model'], candidate, fit, val, cfg, cfg['seeds'][0], True)
            score, per_user = _val_score(np.asarray(answer['prediction']), val)
            trials.append(dict(candidate=cid, params=candidate, score=score, epochs=answer['epochs'],
                               seconds=answer['seconds']))
            if best is None or score < best['score']:
                best = dict(candidate=cid, params=candidate, score=score, epochs=answer['epochs'],
                            dev_per_user=per_user)
        return group, dict(best=best, trials=trials)
    with ThreadPoolExecutor(max_workers=cfg['parallel_jobs']) as pool:
        for group, res in pool.map(lambda kv: one(*kv), groups.items()):
            out[group] = res
            print(f"  tuning {spec['name']} | {group}: candidato {res['best']['candidate']} "
                  f"MASE val={res['best']['score']:.3f} epoche={res['best']['epochs']}", flush=True)
    return out


def run_spec(spec, tuning, panel, users, city, origins_by_user, refit_of, cfg, parts):
    """Refit periodico con configurazione congelata; i parziali completati vengono riusati."""
    parts = Path(parts)
    parts.mkdir(parents=True, exist_ok=True)
    groups = {u: [u] for u in users} if spec['strategy'] == 'locale' else {'all': list(users)}
    seeds = cfg['seeds'] if spec['model'] in ('GRU', 'TFT') else cfg['seeds'][:1]
    jobs = [(r, g, m, s) for r in pd.DatetimeIndex(refit_of.unique()).sort_values()
            for g, m in groups.items() for s in seeds]

    def one(job):
        refit, group, members, seed = job
        path = parts/f"{spec['name']}__{refit:%Y%m%d}__{group}__{seed}.json"
        if path.exists():
            return json.loads(path.read_text())
        todo = {u: [o for o in origins_by_user[u] if refit_of[o] == refit] for u in members}
        todo = {u: v for u, v in todo.items() if v}
        if not todo:
            return []
        best = tuning[group]['best']
        params = dict(best['params'], epochs=max(int(best['epochs']), 1))
        fit, scal = training_samples(panel, members, users, city,
                                     refit-pd.Timedelta(cfg['train_days'], unit='D'), refit, cfg)
        pred_in, actual = prediction_samples(panel, todo, users, city, scal, cfg, False)
        answer = model_job(spec['model'], params, fit, pred_in, cfg, seed, False)
        pred = np.asarray(answer['prediction'])*pred_in['std'][:, None]+pred_in['mean'][:, None]
        rows = [dict(model=spec['name'], seed=int(seed), user=str(u), origin=str(o), pred=p.tolist(),
                     actual=a.tolist()) for u, o, p, a in zip(pred_in['user'], pred_in['origin'], pred, actual)]
        path.write_text(json.dumps(rows))
        print(f"  {spec['name']} | refit {refit.date()} | {group} | seed {seed} | "
              f"{len(rows)} previsioni | {answer['seconds']:.0f}s", flush=True)
        return rows
    rows = []
    with ThreadPoolExecutor(max_workers=cfg['parallel_jobs']) as pool:
        for part in pool.map(one, jobs):
            rows.extend(part)
    for r in rows:
        r['origin'] = pd.Timestamp(r['origin'])
        r['pred'], r['actual'] = np.asarray(r['pred']), np.asarray(r['actual'])
    return rows


# --------------------------------------------------------------------------- combinazioni
def combine_forecasts(rows, model_a, model_b, cfg, window=15, min_past=5, grid=(0., .25, .5, .75, 1.)):
    """Due combinazioni di A (es. LightGBM) e B (es. Naive24), entrambe causali:
    - media semplice 50/50, senza parametri;
    - adattiva per utente: peso di A scelto sulla griglia minimizzando il MAE delle ultime `window` origini
      il cui target è già osservato all'origine corrente; 0.5 finché ce ne sono meno di `min_past`."""
    preds = {}
    for r in rows:
        if r['model'] in (model_a, model_b):
            key = (r['model'], r['user'], pd.Timestamp(r['origin']))
            preds.setdefault(key, []).append((np.asarray(r['pred'], float), np.asarray(r['actual'], float)))
    # Più seed: media delle previsioni (per gli alberi c'è un solo seed).
    preds = {k: (np.mean([p for p, _ in v], axis=0), v[0][1]) for k, v in preds.items()}
    out, h = [], cfg['horizon']*H
    users = sorted({k[1] for k in preds})
    for user in users:
        origins = sorted(o for (m, u, o) in preds if m == model_a and u == user and (model_b, u, o) in preds)
        history = []  # (origine, previsione A, previsione B, osservato)
        for origin in origins:
            a, actual = preds[(model_a, user, origin)]
            b, _ = preds[(model_b, user, origin)]
            past = [x for x in history if x[0]+h <= origin][-window:]
            weight = .5
            if len(past) >= min_past:
                errors = [np.mean([np.abs(w*pa+(1-w)*pb-y).mean() for _, pa, pb, y in past]) for w in grid]
                weight = float(grid[int(np.argmin(errors))])
            out.append(dict(model=f'Comb_media_{model_a}', seed=-1, user=user, origin=origin,
                            pred=.5*a+.5*b, actual=actual))
            out.append(dict(model=f'Comb_adattiva_{model_a}', seed=-1, user=user, origin=origin,
                            pred=weight*a+(1-weight)*b, actual=actual, status=f'peso_{weight:.2f}'))
            history.append((origin, a, b, actual))
    return out


# --------------------------------------------------------------------------- metriche
def scale_table(panel, users, refits, cfg):
    return {(u, r): scaling(panel[u].loc[r-pd.Timedelta(cfg['train_days'], unit='D'):r-H])['scale']
            for u in users for r in refits}


def metrics(rows, scales, refit_of, clip=True):
    """Una riga per modello/seed/utente/origine. Previsioni troncate a zero per tutti i modelli."""
    out = []
    for r in rows:
        p = np.maximum(r['pred'], 0) if clip else r['pred']
        a = r['actual']
        mae = float(np.abs(a-p).mean())
        out.append(dict(model=r['model'], seed=r['seed'], user=r['user'], origin=pd.Timestamp(r['origin']),
                        MAE=mae, RMSE=float(np.sqrt(np.square(a-p).mean())),
                        MASE=mae/scales[(r['user'], refit_of[pd.Timestamp(r['origin'])])],
                        status=r.get('status', 'ok')))
    return pd.DataFrame(out)


def seed_mean(m):
    """Prestazione attesa di un singolo addestramento: media sui seed per utente/origine."""
    return m.groupby(['model', 'user', 'origin'], as_index=False)[['MAE', 'RMSE', 'MASE']].mean()


def _ratio(num, den):
    """Rapporto di MAE; indefinito (NaN) se il riferimento ha errore nullo, es. impianto sempre fermo."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, np.maximum(num, 1e-12)/den, np.nan)


def _geom(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def leaderboard(sm, reference='Naive168'):
    """relMAE e quota di utenti migliorati sono calcolate rispetto al modello `reference`."""
    ref = sm[sm.model == reference].set_index(['user', 'origin']).MAE.rename('MAE_rif')
    sm = sm.join(ref, on=['user', 'origin'])
    per = sm.groupby(['model', 'user']).agg(MASE=('MASE', 'mean'), MAE=('MAE', 'sum'),
                                            MAE_ref=('MAE_rif', 'sum'), RMSE=('RMSE', 'mean'))
    per['relMAE'] = _ratio(per.MAE.to_numpy(), per.MAE_ref.to_numpy())
    board = per.groupby('model').agg(MASE_medio=('MASE', 'mean'), MASE_mediano=('MASE', 'median'),
                                     MASE_peggiore=('MASE', 'max'), relMAE_geom=('relMAE', _geom),
                                     quota_utenti_meglio_rif=('relMAE', lambda x: float((x < 1).mean())),
                                     utenti=('MASE', 'count'))
    return board.sort_values('MASE_medio'), per


def matrix(sm, value='MASE'):
    return sm.groupby(['user', 'model'])[value].mean().unstack('model')


# --------------------------------------------------------------------------- test statistici
def dm_test(loss_a, loss_b, lag=None):
    """Diebold-Mariano su perdite per blocco (h=1 a livello di blocco), varianza HAC di Newey-West
    e correzione di Harvey-Leybourne-Newbold. Stat < 0: A ha perdita minore di B."""
    d = np.asarray(loss_a, float)-np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    lag = int(np.floor(T**(1/3))) if lag is None else lag
    dc = d-d.mean()
    var = dc @ dc/T
    for k in range(1, lag+1):
        var += 2*(1-k/(lag+1))*(dc[k:] @ dc[:-k])/T
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean()/np.sqrt(var/T)*np.sqrt((T-1)/T)
    return float(stat), float(2*stats.t.sf(abs(stat), T-1))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(ok)-rank)*p[i]))
        out[i] = running
    return out


def dm_per_user(sm, reference='Naive168', loss='MAE', alpha=.05):
    """Ogni modello contro il riferimento, utente per utente; Holm sugli utenti dentro ciascun modello."""
    wide = sm.pivot_table(index=['user', 'origin'], columns='model', values=loss)
    rows = []
    for model in wide.columns.drop(reference):
        for user, g in wide.groupby(level='user'):
            g = g[[model, reference]].dropna()
            stat, p = dm_test(g[model], g[reference])
            rows.append(dict(model=model, user=user, origini=len(g), DM=stat, p=p,
                             rel=float(_ratio(g[model].sum(), g[reference].sum()))))
    out = pd.DataFrame(rows)
    out['p_holm'] = out.groupby('model').p.transform(lambda s: holm(s.to_numpy()))
    out['esito'] = np.select([(out.p_holm < alpha) & (out.DM < 0), (out.p_holm < alpha) & (out.DM > 0)],
                             ['meglio', 'peggio'], 'n.s.')
    return out


def dm_panel(sm, alpha=.05):
    """Serie per origine del MASE medio sugli utenti; DM per ogni coppia, Holm su tutte le coppie."""
    series = sm.groupby(['origin', 'model']).MASE.mean().unstack('model')
    models = list(series.mean().sort_values().index)
    rows = []
    for i, a in enumerate(models):
        for b in models[i+1:]:
            g = series[[a, b]].dropna()
            stat, p = dm_test(g[a], g[b])
            rows.append(dict(A=a, B=b, DM=stat, p=p))
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out.p.to_numpy())
    out['significativo'] = out.p_holm < alpha
    return out, series


def friedman_nemenyi(mat, alpha=.05):
    """Ranghi per utente (1 = migliore), Friedman, Iman-Davenport e differenza critica di Nemenyi."""
    mat = mat.dropna()
    N, k = mat.shape
    ranks = mat.rank(axis=1, method='average')
    avg = ranks.mean().sort_values()
    chi2, p = stats.friedmanchisquare(*[mat[c] for c in mat.columns])
    ff = (N-1)*chi2/(N*(k-1)-chi2) if N*(k-1) > chi2 else np.inf
    p_id = float(stats.f.sf(ff, k-1, (k-1)*(N-1)))
    q = stats.studentized_range.ppf(1-alpha, k, np.inf)/np.sqrt(2)
    cd = float(q*np.sqrt(k*(k+1)/(6*N)))
    return dict(avg_ranks=avg, chi2=float(chi2), p_friedman=float(p), F_iman_davenport=float(ff),
                p_iman_davenport=p_id, cd=cd, N=N, k=k)


def wilcoxon_holm(mat, reference):
    mat = mat.dropna()
    rows = []
    for model in mat.columns.drop(reference):
        d = mat[model]-mat[reference]
        try:
            res = stats.wilcoxon(mat[model], mat[reference])
            p = float(res.pvalue)
        except ValueError:
            p = np.nan
        rows.append(dict(model=model, mediana_diff=float(d.median()), utenti_meglio=int((d < 0).sum()),
                         utenti=len(d), p=p))
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out.p.to_numpy())
    return out.sort_values('mediana_diff')


def plot_cd(avg_ranks, cd, ax, title=''):
    """Diagramma delle differenze critiche (Demšar 2006): barre orizzontali = gruppi non distinguibili."""
    names, r = list(avg_ranks.index), avg_ranks.to_numpy()
    k = len(r)
    ax.set_xlim(.5, k+.5)
    ax.set_ylim(-1.5-k/2, 1.5)
    ax.hlines(0, 1, k, color='k')
    for t in range(1, k+1):
        ax.vlines(t, 0, .12, color='k')
        ax.text(t, .25, str(t), ha='center')
    ax.hlines(1, 1, 1+cd, color='tab:red', lw=3)
    ax.text(1+cd/2, 1.2, f'CD = {cd:.2f}', ha='center', color='tab:red')
    half = (k+1)//2
    for i, (n, v) in enumerate(zip(names, r)):
        y = -.6-.45*(i if i < half else k-1-i)
        x_text = .55 if i < half else k+.45
        ax.plot([v, v, x_text], [0, y, y], color='k', lw=.8)
        ax.text(x_text, y, f'{n} ({v:.2f})', ha='left' if i < half else 'right', va='center',
                fontsize=8, bbox=dict(fc='white', ec='none', pad=.5))
    level, last = 0, -1
    for i in range(k):
        j = max(j for j in range(i, k) if r[j]-r[i] < cd)
        if j > i and j > last:  # solo gruppi massimali
            ax.hlines(-.25-.15*level, r[i]-.03, r[j]+.03, color='tab:blue', lw=4)
            level, last = level+1, j
    ax.axis('off')
    ax.set_title(title)


def block_bootstrap_relmae(sm, reference='Naive168', n_boot=2000, block=7, seed=0):
    """IC 95% della media geometrica sugli utenti di MAE_modello/MAE_riferimento.
    Si ricampionano blocchi di origini consecutive, uguali per tutti gli utenti e modelli."""
    rng = np.random.default_rng(seed)
    origins = np.sort(sm.origin.unique())
    T = len(origins)
    models = sorted(sm.model.unique())
    cube = {m: sm[sm.model == m].pivot(index='user', columns='origin', values='MAE').reindex(columns=origins).to_numpy()
            for m in models}
    ref = cube[reference]
    starts = np.arange(max(T-block+1, 1))
    draws = {m: [] for m in models if m != reference}
    for _ in range(n_boot):
        idx = (rng.choice(starts, int(np.ceil(T/block)))[:, None]+np.arange(block)).ravel()[:T]
        idx = np.minimum(idx, T-1)
        base = np.nansum(ref[:, idx], axis=1)
        for m in draws:
            draws[m].append(_geom(_ratio(np.nansum(cube[m][:, idx], axis=1), base)))
    rows = []
    for m in draws:
        point = _geom(_ratio(np.nansum(cube[m], axis=1), np.nansum(ref, axis=1)))
        lo, hi = np.nanpercentile(draws[m], [2.5, 97.5])
        rows.append(dict(model=m, relMAE_geom=point, IC95_inf=lo, IC95_sup=hi, sotto_1=hi < 1, sopra_1=lo > 1))
    return pd.DataFrame(rows).sort_values('relMAE_geom')


# --------------------------------------------------------------------------- selezione dell'utente
def selection_bias(dev_skill, test_skill, focus=None):
    """Quanto è ottimistico valutare solo l'utente migliore in sviluppo?
    dev_skill/test_skill: DataFrame utenti x modelli di skill rispetto al riferimento (1 - rapporto MAE)."""
    rows = []
    for model in test_skill.columns.intersection(dev_skill.columns):
        d, t = dev_skill[model].dropna(), test_skill[model].dropna()
        common = d.index.intersection(t.index)
        d, t = d[common], t[common]
        if len(common) < 3:
            continue
        chosen = d.idxmax()
        rho, p = stats.spearmanr(d, t)
        row = dict(model=model, utenti=len(common), skill_test_pannello=float(t.mean()),
                   skill_test_mediana=float(t.median()), utente_migliore_sviluppo=chosen,
                   skill_test_utente_scelto=float(t[chosen]),
                   ottimismo=float(t[chosen]-t.mean()), spearman_sviluppo_test=float(rho), p_spearman=float(p))
        if focus is not None and focus in common:
            row[f'skill_test_{focus}'] = float(t[focus])
            row[f'rango_{focus}'] = int(t.rank(ascending=False)[focus])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Usare --worker DIRECTORY')
    _worker(sys.argv[2])

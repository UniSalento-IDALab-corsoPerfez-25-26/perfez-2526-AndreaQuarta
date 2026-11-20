"""Motore del benchmark C24. Modelli in processi isolati; nessun training all'import."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import os
import time
import numpy as np
import pandas as pd


def load_panel(root):
    folder = Path(root)/'EWELD/Electricity Consumption/C/C24 Manufacture of basic metals'
    panel = {}
    for path in sorted(folder.glob('*.csv')):
        data = pd.read_csv(path)
        idx = pd.DatetimeIndex(pd.to_datetime(data.Time))
        if idx.has_duplicates or not ((idx.minute % 15 == 0) & (idx.second == 0)).all():
            raise ValueError(f'Timestamp non validi: {path}')
        s = pd.Series(pd.to_numeric(data.Value, errors='raise').to_numpy(), index=idx).sort_index()
        if (s.dropna() < 0).any():
            raise ValueError(f'Consumi negativi: {path}')
        panel[path.stem] = s.replace([np.inf, -np.inf], np.nan).resample('h').sum(min_count=4)
    if not panel:
        raise ValueError('Nessuna utenza C24 trovata.')
    return panel


def locations(root, users):
    result = {}
    for k in (1, 2, 3):
        for user in pd.read_csv(Path(root)/f'EWELD/User Location/U_CT{k}.csv').iloc[:, 0].astype(str).str.strip():
            if user in result:
                raise ValueError(f'Localizzazione duplicata: {user}')
            result[user] = k
    if not set(users) <= set(result):
        raise ValueError('Localizzazione mancante per alcune utenze.')
    return result


def coverage(panel, origins, cfg):
    """Criterio solo di disponibilità, non dipende da livello o prevedibilità."""
    rows = []
    for user, series in panel.items():
        for origin in map(pd.Timestamp, origins):
            index = pd.date_range(origin-pd.Timedelta(cfg['train_days'], unit='D'),
                                  periods=cfg['train_days']*24+cfg['horizon'], freq='h')
            missing = int(series.reindex(index).isna().sum())
            rows.append(dict(user=user, origin=str(origin), missing_hours=missing, eligible=missing == 0))
    return pd.DataFrame(rows)


def calendar(index):
    hour = index.hour.to_numpy()
    week = index.dayofweek.to_numpy()*24+hour
    day = index.dayofyear.to_numpy()
    holiday = np.zeros(len(index))
    for date in ['2016-02-08','2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12','2022-02-01']:
        holiday = np.maximum(holiday, ((index >= pd.Timestamp(date)-pd.Timedelta(7, unit='D')) &
                                      (index <= pd.Timestamp(date)+pd.Timedelta(10, unit='D'))).astype(float))
    return np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*week/168),np.cos(2*np.pi*week/168),
        np.sin(2*np.pi*day/366),np.cos(2*np.pi*day/366),holiday]).astype('float32')


def load_weather(root):
    result = {}
    for k in (1, 2, 3):
        df = pd.read_csv(Path(root)/f'EWELD/Weather Data/W{k}.csv')
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop('Time')))
        if df.index.has_duplicates:
            raise ValueError('Meteo duplicato.')
        cols = ['Temperature(F)','Dew Point(F)','Humidity(%)','Wind Speed(mph)','Wind Gust(mph)','Pressure(in)']
        x = df[cols].apply(pd.to_numeric, errors='coerce').sort_index()
        x[cols[:2]] = (x[cols[:2]]-32)*5/9
        x[cols[3:5]] *= .44704
        x[cols[5]] *= 33.8638866667
        hourly = x.resample('h').mean().where(x.resample('h').count().eq(4))
        # Solo osservazioni storiche, ritardo prudenziale per l'interpolazione originaria.
        result[k] = hourly.shift(1).replace([np.inf,-np.inf],np.nan)
    return result


def scaling(history):
    a = history.to_numpy(float)
    mean, std = a.mean(), a.std()
    std = max(std, 1e-6)
    raw = float(np.abs(a[24:]-a[:-24]).mean())
    floor = max(1e-6, .001*float(np.abs(a).mean()))
    return dict(mean=mean, std=std, raw_scale=raw, scale=max(raw, floor), scale_floored=raw < floor)


def assign_groups(histories, strategy, cfg):
    users = sorted(histories)
    if strategy == 'locale':
        return {u: u for u in users}
    if strategy == 'globale':
        return {u: 'all' for u in users}
    if strategy != 'gruppo':
        raise ValueError(strategy)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    descriptors = []
    for u in users:
        s = histories[u]
        scale = max(s.abs().mean(), 1e-6)
        daily = s.groupby(s.index.hour).mean().reindex(range(24), fill_value=0).to_numpy()/scale
        weekly = s.groupby(s.index.dayofweek).mean().reindex(range(7), fill_value=0).to_numpy()/scale
        descriptors.append(np.r_[daily, weekly, s.std()/scale, (s < .05*scale).mean()])
    x = StandardScaler().fit_transform(descriptors)
    n = min(cfg['clusters'], len(users), len(np.unique(x, axis=0)))
    labels = KMeans(n_clusters=n, n_init=10, random_state=cfg['cluster_seed']).fit_predict(x)
    return dict(zip(users, map(int, labels)))


def make_samples(histories, contexts, origins, users, all_users, city, weather, with_weather, cfg, training=False):
    """histories: solo fit per scaler; contexts: valori disponibili; origins: target start."""
    h, lookback = cfg['horizon'], cfg['lookback']
    xs, fs, ys, statics, scales, means, stds, ids, dates = [], [], [], [], [], [], [], [], []
    for user in sorted(users):
        hist = histories[user]
        params = scaling(hist)
        s = contexts[user]
        if training:
            positions = np.arange(lookback, len(hist)-h+1, cfg['stride'])
            if len(positions) > cfg['max_samples_per_user']:
                positions = positions[np.linspace(0,len(positions)-1,cfg['max_samples_per_user'],dtype=int)]
            user_origins = hist.index[positions]
        else:
            user_origins = pd.DatetimeIndex(origins)
        known_city = np.eye(3, dtype='float32')[city[user]-1]
        identity = np.eye(len(all_users), dtype='float32')[all_users.index(user)]
        extra = None
        if with_weather:
            raw = weather[city[user]]
            filled = raw.ffill(limit=6)
            fit_w = filled.reindex(hist.index)
            med = fit_w.median().fillna(0)
            mu = fit_w.fillna(med).mean()
            sd = fit_w.fillna(med).std(ddof=0).clip(lower=1e-6)
            extra = pd.concat([(filled.fillna(med)-mu)/sd, raw.isna().astype(float)], axis=1)
        for origin in user_origins:
            past_idx = pd.date_range(origin-pd.Timedelta(lookback, unit='h'), periods=lookback, freq='h')
            future_idx = pd.date_range(origin, periods=h, freq='h')
            past = s.reindex(past_idx)
            target = s.reindex(future_idx)
            if past.isna().any() or (training and target.isna().any()):
                raise ValueError(f'Finestra incompleta {user} {origin}')
            parts = [((past.to_numpy()-params['mean'])/params['std'])[:,None], calendar(past_idx)]
            if extra is not None:
                ex = extra.reindex(past_idx).to_numpy()
                if not np.isfinite(ex).all():
                    raise ValueError(f'Meteo senza copertura {user} {origin}')
                parts.append(ex)
            xs.append(np.column_stack(parts))
            fs.append(calendar(future_idx))
            # In inferenza si possono omettere i target: i modelli non li leggono.
            ys.append((target.to_numpy()-params['mean'])/params['std'])
            statics.append(np.r_[identity,known_city])
            means.append(params['mean']);stds.append(params['std']);scales.append(params['scale'])
            ids.append(user);dates.append(str(origin))
    arrays = dict(x=np.asarray(xs,dtype='float32'), f=np.asarray(fs,dtype='float32'),
                  y=np.asarray(ys,dtype='float32'), static=np.asarray(statics,dtype='float32'),
                  mean=np.asarray(means),std=np.asarray(stds),scale=np.asarray(scales),
                  user=np.asarray(ids),origin=np.asarray(dates))
    if not xs:
        raise ValueError('Nessun esempio di training: ampliare lo storico.')
    if training:
        # Peso totale uguale per impianto, anche se i conteggi dovessero differire.
        counts = pd.Series(ids).value_counts()
        weight = np.array([1/counts[u] for u in ids]);weight /= weight.mean()
        arrays['weight'] = weight.astype('float32')
    return arrays


def model_job(model, cfg_model, fit, predict, cfg, seed, tune):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='eweld_multi_') as temp:
        temp = Path(temp)
        for label, data in [('fit',fit),('predict',predict)]:
            np.savez(temp/f'{label}.npz', **data)
        request = dict(model=model, model_config=cfg_model, seed=int(seed), tune=tune,
                       threads=cfg['threads'], epochs=cfg['max_epochs'], patience=cfg['patience'],
                       batch_size=cfg['batch_size'])
        (temp/'request.json').write_text(json.dumps(request))
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            env[key] = str(cfg['threads'])
        env.pop('KMP_DUPLICATE_LIB_OK',None)
        process = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(temp)],
                                 env=env,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError(f'{model} fallito: {process.stderr[-4000:]} {process.stdout[-2000:]}')
        result = json.loads((temp/'result.json').read_text())
        result['seconds'] = time.perf_counter()-start
        return result


def evaluate(actual, predicted, history, naive):
    a,p = np.asarray(actual),np.asarray(predicted)
    if a.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError('Target o previsione non validi.')
    q=scaling(history);mae=float(np.abs(a-p).mean());base=float(np.abs(a-naive).mean())
    # Regimi descrittivi dedotti dal consumo; soglia appresa solo sul train.
    positive=history[history>0]
    threshold=.05*float(positive.median()) if len(positive) else 0.
    low=a<=threshold
    previous=np.r_[float(history.iloc[-1])<=threshold,low[:-1]]
    transition=low!=previous
    return dict(MAE_low=float(np.abs(a[low]-p[low]).mean()) if low.any() else np.nan,
                MAE_transition=float(np.abs(a[transition]-p[transition]).mean()) if transition.any() else np.nan,
                low_hours=int(low.sum()),transition_hours=int(transition.sum()),
                MAE=mae,RMSE=float(np.sqrt(np.square(a-p).mean())),
                MASE=mae/q['raw_scale'] if q['raw_scale']>0 else np.nan,
                MASE_stabilizzato=mae/q['scale'],scale_floored=q['scale_floored'],
                MAE_naive168=base,skill_naive168=1-mae/base if base>1e-12 else np.nan,
                beats_naive168=bool(mae<base),negative_predictions=int((p<0).sum()))


def run_benchmark(panel, city, weather, users, origins, specs, cfg, output):
    """Walk-forward esterna; tuning interno separato a ogni origine, refit, previsione."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[];trials=[];forecasts=[];memberships=[]
    users=sorted(users)
    for origin in map(pd.Timestamp,origins):
        start=origin-pd.Timedelta(cfg['train_days'], unit='D')
        split=origin-pd.Timedelta(cfg['val_blocks']*cfg['horizon'], unit='h')
        fit_hist={u:panel[u].loc[start:split-pd.Timedelta(1, unit='h')] for u in users}
        full_hist={u:panel[u].loc[start:origin-pd.Timedelta(1, unit='h')] for u in users}
        inner_origins=pd.date_range(split, periods=cfg['val_blocks'], freq=f"{cfg['horizon']}h")
        # Baseline valutate sugli stessi utenti e origini; nessuna selezione sul test.
        for user in users:
            actual=panel[user].reindex(pd.date_range(origin,periods=cfg['horizon'],freq='h')).to_numpy()
            n168=np.resize(full_hist[user].iloc[-168:].to_numpy(),cfg['horizon'])
            for name, pred in [('Naive168',n168),('Naive24',np.resize(full_hist[user].iloc[-24:].to_numpy(),cfg['horizon']))]:
                row=dict(spec=name,user=user,origin=str(origin),seed=-1,seconds=0.,**evaluate(actual,pred,full_hist[user],n168))
                results.append(row)
                forecasts.extend(dict(spec=name,user=user,origin=str(origin),seed=-1,horizon=k+1,actual=float(a),prediction=float(p))
                                 for k,(a,p) in enumerate(zip(actual,pred)))
        for spec in specs:
            grouping=assign_groups(fit_hist,spec['strategy'],cfg)
            memberships.extend(dict(spec=spec['name'],origin=str(origin),user=u,group=str(gr)) for u,gr in grouping.items())
            for seed in cfg['seeds']:
                for group in sorted(set(grouping.values()),key=str):
                    members=[u for u in users if grouping[u]==group]
                    print(f"{origin.date()} | {spec['name']} | seed {seed} | gruppo {group} | {len(members)} utenti",flush=True)
                    args=(members,users,city,weather,spec['weather'],cfg)
                    fit=make_samples(fit_hist,fit_hist,None,*args,training=True)
                    val=make_samples(fit_hist,full_hist,inner_origins,*args)
                    best=None
                    for cid,parameters in enumerate(spec['grid']):
                        answer=model_job(spec['model'],parameters,fit,val,cfg,seed,True)
                        pred=np.asarray(answer['prediction'])
                        losses=np.mean(np.abs(pred-val['y'])*val['std'][:,None],axis=1)/val['scale']
                        score=float(pd.DataFrame({'user':val['user'],'loss':losses}).groupby('user').loss.mean().mean())
                        record=dict(spec=spec['name'],origin=str(origin),seed=seed,group=str(group),candidate=cid,
                                    score=score,epochs=answer['epochs'],seconds=answer['seconds'],parameters=parameters,
                                    history=answer.get('history',[]))
                        trials.append(record)
                        if best is None or score<best[0]:best=(score,parameters,answer['epochs'],cid)
                        print(f'  candidato {cid}: MASE stabilizzato val={score:.4f}; epoche={answer["epochs"]}',flush=True)
                    final_cfg=dict(best[1],epochs=best[2])
                    final_fit=make_samples(full_hist,full_hist,None,*args,training=True)
                    # Contesto solo passato: target futuri sostituiti da NaN e mai inviati reali al worker.
                    test=make_samples(full_hist,full_hist,[origin],*args)
                    answer=model_job(spec['model'],final_cfg,final_fit,test,cfg,seed,False)
                    pred=np.asarray(answer['prediction'])*test['std'][:,None]+test['mean'][:,None]
                    for j,user in enumerate(test['user']):
                        actual=panel[user].reindex(pd.date_range(origin,periods=cfg['horizon'],freq='h')).to_numpy()
                        naive=np.resize(full_hist[user].iloc[-168:].to_numpy(),cfg['horizon'])
                        results.append(dict(spec=spec['name'],origin=str(origin),user=user,seed=seed,
                            seconds=answer['seconds']/len(members),selected_candidate=best[3],
                            **evaluate(actual,pred[j],full_hist[user],naive)))
                        forecasts.extend(dict(spec=spec['name'],origin=str(origin),user=user,seed=seed,horizon=k+1,
                                              actual=float(a),prediction=float(p)) for k,(a,p) in enumerate(zip(actual,pred[j])))
                    # Snapshot dopo ogni gruppo completato. Un errore interrompe: niente classifiche parziali silenziose.
                    pd.DataFrame(results).to_csv(output/'metriche_parziali.csv',index=False)
                    (output/'ricerche.json').write_text(json.dumps(trials,indent=2))
        pd.DataFrame(forecasts).to_csv(output/'previsioni.csv',index=False)
    metrics=pd.DataFrame(results)
    metrics.to_csv(output/'metriche.csv',index=False)
    pd.DataFrame(memberships).to_csv(output/'gruppi.csv',index=False)
    (output/'configurazione.json').write_text(json.dumps(dict(config=cfg,specs=specs,users=users,origins=list(map(str,origins))),indent=2))
    return metrics,pd.DataFrame(forecasts),trials


def leaderboard(metrics):
    # Prima media seed/origini per utente, poi media utenti con uguale peso.
    per=metrics.groupby(['spec','user']).agg(MASE_stabilizzato=('MASE_stabilizzato','mean'),
        RMSE=('RMSE','mean'),MAE=('MAE','mean'),skill_naive168=('skill_naive168','mean'),
        beats_naive168=('beats_naive168','mean'))
    out=per.groupby('spec').agg(score=('MASE_stabilizzato','mean'),
        median_user=('MASE_stabilizzato','median'),worst_user=('MASE_stabilizzato','max'),
        users=('MASE_stabilizzato','count'),win_fraction=('beats_naive168','mean'))
    return out.sort_values('score'),per


# Tutti gli import dei backend sono confinati al processo worker.
def _worker(folder):
    folder=Path(folder);request=json.loads((folder/'request.json').read_text())
    with np.load(folder/'fit.npz') as file:fit={k:file[k] for k in file.files}
    with np.load(folder/'predict.npz') as file:val={k:file[k] for k in file.files}
    cfg=request['model_config'];name=request['model'];seed=request['seed'];tune=request['tune']
    epochs=0;history=[]
    if name=='XGBoost':
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        def features(data):
            x=data['x'];parts=[x[:,:,0],x[:,-1,1:],data['f'].reshape(len(x),-1),data['static']]
            for width in (24,168):
                for fn in (np.mean,np.std,np.min,np.max):parts.append(fn(x[:,-width:,:],axis=1))
            return np.concatenate(parts,axis=1)
        args={k:v for k,v in cfg.items() if k!='epochs'}
        model=MultiOutputRegressor(XGBRegressor(**args,objective='reg:squarederror',tree_method='hist',
                                    n_jobs=request['threads'],random_state=seed),n_jobs=1)
        model.fit(features(fit),fit['y'],sample_weight=fit['weight'])
        prediction=model.predict(features(val))
    else:
        import random
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset,DataLoader
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(request['threads'])
        batch_size=request['batch_size'];h=fit['f'].shape[1];lookback=fit['x'].shape[1]
        class GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=nn.GRU(fit['x'].shape[2],cfg['hidden'],num_layers=cfg['layers'],batch_first=True,
                                    dropout=.1 if cfg['layers']>1 else 0.)
                self.head=nn.Sequential(nn.Linear(cfg['hidden']+fit['f'].shape[2]+fit['static'].shape[1]+1,cfg['hidden']),
                                        nn.ReLU(),nn.Linear(cfg['hidden'],1))
            def forward(self,x,f,static):
                _,z=self.encoder(x);z=z[-1][:,None,:].expand(-1,h,-1)
                pos=torch.linspace(1/h,1,h).view(1,h,1).expand(len(x),-1,-1)
                return self.head(torch.cat([z,f,static[:,None,:].expand(-1,h,-1),pos],dim=-1)).squeeze(-1)
        is_tft=name=='TFT'
        if name not in ('GRU','TFT'):raise ValueError(name)
        if is_tft:
            from pytorch_forecasting import TimeSeriesDataSet,TemporalFusionTransformer
            from pytorch_forecasting.metrics import RMSE
            # Ogni esempio è un blocco sintetico autonomo. Il filtro impedisce finestre a cavallo dei blocchi.
            def frame(data):
                frames=[];counter={};length=lookback+h
                for j,user in enumerate(data['user']):
                    offset=counter.get(user,0)*length;counter[user]=counter.get(user,0)+1
                    target=np.r_[data['x'][j,:,0],np.nan_to_num(data['y'][j],nan=0.)]
                    f=pd.DataFrame({'plant':str(user),'time_idx':np.arange(offset,offset+length),'target':target})
                    for k in range(data['f'].shape[2]):
                        f[f'known_{k}']=np.r_[data['x'][j,:,1+k],data['f'][j,:,k]]
                    for k in range(1+data['f'].shape[2],data['x'].shape[2]):
                        f[f'observed_{k}']=np.r_[data['x'][j,:,k],np.zeros(h)]
                    for k in range(data['static'].shape[1]):f[f'static_{k}']=data['static'][j,k]
                    frames.append(f)
                return pd.concat(frames,ignore_index=True)
            train_frame=frame(fit);predict_frame=frame(val)
            known=[c for c in train_frame if c.startswith('known_')]
            observed=[c for c in train_frame if c.startswith('observed_')]
            static=[c for c in train_frame if c.startswith('static_')]
            dataset=TimeSeriesDataSet(train_frame,time_idx='time_idx',target='target',group_ids=['plant'],
                min_encoder_length=lookback,max_encoder_length=lookback,min_prediction_length=h,max_prediction_length=h,
                static_reals=static,time_varying_known_reals=known,time_varying_unknown_reals=['target']+observed,
                target_normalizer=None,add_relative_time_idx=True)
            select=lambda index:index.time_idx_first_prediction.mod(lookback+h).eq(lookback)
            dataset=dataset.filter(select)
            prediction_ds=TimeSeriesDataSet.from_dataset(dataset,predict_frame,stop_randomization=True).filter(select)
            assert len(dataset)==len(fit['x']) and len(prediction_ds)==len(val['x'])
            # Ordine costruito per utente e origine; verificato anche dal decoded_index.
            assert dataset.decoded_index['plant'].tolist()==fit['user'].tolist()
            assert prediction_ds.decoded_index['plant'].tolist()==val['user'].tolist()
            train_loader=dataset.to_dataloader(train=True,batch_size=batch_size,num_workers=0,drop_last=False)
            pred_loader=prediction_ds.to_dataloader(train=False,batch_size=batch_size,num_workers=0)
            model=TemporalFusionTransformer.from_dataset(dataset,hidden_size=cfg['hidden'],attention_head_size=2,
                    lstm_layers=cfg['layers'],hidden_continuous_size=cfg['hidden'],dropout=.1,output_size=1,
                    loss=RMSE(),mask_bias=-1e4,log_interval=-1)
            def forward(batch):return model(batch[0]).prediction.squeeze(-1),batch[1][0]
        else:
            def loader(data,shuffle):
                tensors=[torch.from_numpy(data[k]).float() for k in ('x','f','static','y')]
                return DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=shuffle,
                                  generator=torch.Generator().manual_seed(seed))
            train_loader=loader(fit,True);pred_loader=loader(val,False);model=GRU()
            def forward(batch):return model(*batch[:3]),batch[3]
        # I campioni per utente sono uguali per copertura fissa e stride comune.
        counts=pd.Series(fit['user']).value_counts()
        assert counts.nunique()==1,'Reti: richiesto bilanciamento esatto dei campioni tra utenti.'
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.get('lr',.001))
        limit=request['epochs'] if tune else cfg['epochs']
        best=float('inf');best_weights=None;stale=0
        def predict():
            model.eval();out=[]
            with torch.no_grad():
                for batch in pred_loader:out.append(forward(batch)[0].numpy())
            return np.concatenate(out)
        for epoch in range(1,limit+1):
            model.train();total=0.;count=0
            for batch in train_loader:
                optimizer.zero_grad();out,target=forward(batch)
                loss=nn.functional.mse_loss(out,target)
                if not torch.isfinite(loss):raise ValueError('Loss non finita')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                total+=loss.item()*len(target);count+=len(target)
            score=None
            if tune:
                p=predict();losses=np.abs(p-val['y']).mean(axis=1)*val['std']/val['scale']
                score=float(pd.DataFrame({'u':val['user'],'s':losses}).groupby('u').s.mean().mean())
                if score<best:
                    best=score;epochs=epoch;stale=0;best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:stale+=1
            else:epochs=epoch
            history.append(dict(epoch=epoch,train_loss=total/count,validation_score=score))
            if tune and stale>=request['patience']:break
        if tune:
            if best_weights is None:raise ValueError('Nessuna epoca valida')
            model.load_state_dict(best_weights)
        prediction=predict()
    if prediction.shape != val['y'].shape or not np.isfinite(prediction).all():raise ValueError('Output non valido')
    (folder/'result.json').write_text(json.dumps(dict(prediction=prediction.tolist(),epochs=epochs,history=history)))


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--worker':raise SystemExit('Usare --worker DIRECTORY')
    _worker(sys.argv[2])

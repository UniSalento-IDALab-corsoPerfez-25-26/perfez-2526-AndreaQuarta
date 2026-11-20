"""Metriche di tutti i benchmark del progetto (c24_benchmark, lungo_termine_benchmark, …).

Unico file delle metriche: ogni notebook e ogni libreria importa da qui, così le
stesse colonne sono calcolate dallo stesso codice ovunque. Cambia solo, da un
benchmark all'altro, la scala del MASE (la passa chi chiama) e l'unità di misura
(vedi `descrizione`).

Gli array sono (n_origini, H): una riga per origine, una colonna per passo
dell'orizzonte. Le metriche sono calcolate sull'insieme di tutti i passi di tutte
le origini; l'errore sul totale somma ogni riga e fa la media sulle origini.
Il test di Diebold-Mariano (`confronto_dm`) usa invece una perdita per origine.
"""
import numpy as np
import pandas as pd

ORDINE = ['MAE', 'RMSE', 'MAPE %', 'sMAPE %', 'MASE', 'Bias',
          'Errore totale %', 'Copertura 80%', 'Larghezza 80%']

DESCRIZIONE = {
    'MAE': 'errore assoluto medio (kWh)',
    'RMSE': 'radice dell’errore quadratico medio (kWh)',
    'MAPE %': 'errore percentuale assoluto medio sui valori reali non nulli (per il confronto con la letteratura)',
    'sMAPE %': 'errore percentuale simmetrico',
    'MASE': 'MAE / MAE del naive stagionale sul train (<1 batte il naive)',
    'Bias': 'media di (previsto − reale): >0 sovrastima (kWh)',
    'Errore totale %': 'errore % sul consumo cumulato di ogni finestra, mediato sulle origini',
    'Copertura 80%': '% di valori reali dentro l’intervallo q10–q90 (ideale 80; vuota senza quantili)',
    'Larghezza 80%': 'ampiezza media dell’intervallo q10–q90 (kWh; vuota senza quantili)',
}

FORMATI = {'MAE': '{:.1f}', 'RMSE': '{:.1f}', 'MAPE %': '{:.2f}', 'sMAPE %': '{:.2f}', 'MASE': '{:.3f}',
           'Bias': '{:+.1f}', 'Errore totale %': '{:.2f}', 'Copertura 80%': '{:.2f}',
           'Larghezza 80%': '{:.1f}'}

DA_MINIMIZZARE = ['MAE', 'RMSE', 'MAPE %', 'sMAPE %', 'MASE', 'Errore totale %']


def descrizione(unita='kWh', naive='naive stagionale'):
    """DESCRIZIONE con l'unità del benchmark (es. 'kWh/giorno') e il nome del naive che scala il MASE."""
    d = {k: v.replace('(kWh', f'({unita}') for k, v in DESCRIZIONE.items()}
    d['MASE'] = f'MAE / MAE del {naive} sul train (<1 batte il naive)'
    return d


def calcola(reale, previsto, scala_mase, q10=None, q90=None):
    """Tutte le metriche di ORDINE; copertura e larghezza sono NaN se mancano i quantili.

    Array (n, H) o (n,) (un passo per origine).
    """
    a, p = np.asarray(reale, float), np.asarray(previsto, float)
    a, p = a.reshape(len(a), -1), p.reshape(len(p), -1)
    if a.shape != p.shape:
        raise ValueError(f'Dimensioni diverse: reale {a.shape}, previsto {p.shape}')
    if not np.isfinite(p).all():
        raise ValueError('Previsione con valori non finiti.')
    errore = p-a
    denom = np.abs(a)+np.abs(p)
    smape = np.divide(2*np.abs(errore), denom, out=np.zeros_like(errore), where=denom != 0)
    non_nulli = a != 0
    mape = 100*(np.abs(errore[non_nulli])/np.abs(a[non_nulli])).mean() if non_nulli.any() else np.nan
    somma_reale = a.sum(axis=1)
    errore_totale = np.divide(np.abs(p.sum(axis=1)-somma_reale), somma_reale,
                              out=np.full_like(somma_reale, np.nan), where=somma_reale != 0)
    copertura = larghezza = np.nan
    if q10 is not None and q90 is not None:
        q10, q90 = np.asarray(q10, float), np.asarray(q90, float)
        q10, q90 = q10.reshape(len(q10), -1), q90.reshape(len(q90), -1)
        if q10.shape != a.shape or q90.shape != a.shape:
            raise ValueError(f'Quantili di dimensioni diverse dal reale {a.shape}')
        copertura = 100*((q10 <= a) & (a <= q90)).mean()
        larghezza = (q90-q10).mean()
    return {
        'MAE': np.abs(errore).mean(),
        'RMSE': np.sqrt((errore**2).mean()),
        'MAPE %': mape,
        'sMAPE %': 100*smape.mean(),
        'MASE': np.abs(errore).mean()/scala_mase if scala_mase > 0 else np.nan,
        'Bias': errore.mean(),
        'Errore totale %': 100*errore_totale.mean(),
        'Copertura 80%': copertura,
        'Larghezza 80%': larghezza,
    }


def _calcola_previsione(reale, previsione, scala_mase, taglio=None):
    """previsione: array del punto oppure dict con 'punto' ed eventuali 'q10', 'q90'; prime `taglio` colonne."""
    if not isinstance(previsione, dict):
        previsione = {'punto': previsione}

    def primi(x):
        x = np.asarray(x, float)
        x = x.reshape(len(x), -1)
        return x if taglio is None else x[:, :taglio]
    quantili = 'q10' in previsione and 'q90' in previsione
    return calcola(primi(reale), primi(previsione['punto']), scala_mase,
                   q10=primi(previsione['q10']) if quantili else None,
                   q90=primi(previsione['q90']) if quantili else None)


def tabella(previsioni, reale, scala_mase, tagli=None):
    """previsioni: {modello: array (n, H) oppure {'punto', eventuali 'q10', 'q90'}}.

    Senza `tagli` una riga per modello; con `tagli` (es. (30, 60, 90)) una riga per
    (modello, taglio), con le metriche sulle prime `taglio` colonne dell'orizzonte.
    """
    if tagli is None:
        righe = {nome: _calcola_previsione(reale, p, scala_mase) for nome, p in previsioni.items()}
        tab = pd.DataFrame(righe).T[ORDINE]
        tab.index.name = 'modello'
        return tab
    righe = {(nome, taglio): _calcola_previsione(reale, p, scala_mase, taglio)
             for nome, p in previsioni.items() for taglio in tagli}
    tab = pd.DataFrame(righe).T[ORDINE]
    tab.index = pd.MultiIndex.from_tuples(tab.index, names=['modello', 'taglio'])
    return tab


def mae_per_orizzonte(reale, punto):
    """MAE per ciascun passo dell'orizzonte (H,), mediato sulle origini."""
    return np.abs(np.asarray(punto, float)-np.asarray(reale, float)).mean(axis=0)


def diebold_mariano(perdita_a, perdita_b, h=1):
    """Test di Diebold-Mariano con la correzione di Harvey, Leybourne e Newbold (1997).

    perdita_a, perdita_b: perdite per origine (n,), una per finestra di previsione.
    h: le perdite sono autocorrelate fino al ritardo h-1 (finestre sovrapposte).
    Restituisce (statistica, p bilaterale) con la t di Student a n-1 gradi di libertà;
    statistica < 0 vuol dire che `a` sbaglia meno di `b`. NaN se n <= h o se la varianza
    stimata non è positiva (per esempio perdite identiche).
    """
    from scipy import stats
    d = np.asarray(perdita_a, float)-np.asarray(perdita_b, float)
    n = len(d)
    if h < 1 or n <= h:
        return np.nan, np.nan
    scarti = d-d.mean()
    autocov = [(scarti[k:]*scarti[:n-k]).sum()/n for k in range(h)]
    varianza = (autocov[0]+2*sum(autocov[1:]))/n
    if not varianza > 0:
        return np.nan, np.nan
    statistica = d.mean()/np.sqrt(varianza)*np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return statistica, 2*stats.t.sf(abs(statistica), n-1)


def confronto_dm(previsioni, reale, riferimento, tagli, passo=30):
    """Diebold-Mariano di ogni modello contro `riferimento`, per taglio dell'orizzonte.

    previsioni: {modello: array (n, H) oppure {'punto', …}}, con `riferimento` tra le chiavi.
    La perdita di un'origine è il MAE sulle sue prime `taglio` colonne. Con origini distanti circa
    `passo` giorni le finestre lunghe `taglio` si sovrappongono: h = ceil(taglio/passo).
    Indice (modello, taglio); colonne h, DM (< 0: il modello batte il riferimento) e p.
    """
    def punto(p):
        return np.asarray(p['punto'] if isinstance(p, dict) else p, float)
    reale = np.asarray(reale, float)
    base = punto(previsioni[riferimento])
    righe = {}
    for nome, p in previsioni.items():
        if nome == riferimento:
            continue
        for taglio in tagli:
            perdita = np.abs(punto(p)[:, :taglio]-reale[:, :taglio]).mean(axis=1)
            perdita_rif = np.abs(base[:, :taglio]-reale[:, :taglio]).mean(axis=1)
            h = int(np.ceil(taglio/passo))
            statistica, p_valore = diebold_mariano(perdita, perdita_rif, h)
            righe[(nome, taglio)] = {'h': h, 'DM': statistica, 'p': p_valore}
    tab = pd.DataFrame.from_dict(righe, orient='index', columns=['h', 'DM', 'p'])
    tab.index = pd.MultiIndex.from_tuples(tab.index, names=['modello', 'taglio'])
    return tab


def formatta(tab):
    """Formati comuni; in grassetto il minimo delle metriche di errore presenti nella tabella."""
    return tab.style.format({k: v for k, v in FORMATI.items() if k in tab.columns}).highlight_min(
        subset=[c for c in DA_MINIMIZZARE if c in tab.columns], props='font-weight:bold')

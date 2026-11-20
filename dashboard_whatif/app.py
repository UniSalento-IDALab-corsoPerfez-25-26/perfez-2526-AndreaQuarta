"""Dashboard what-if (BOZZA): come cambiano previsione ed errore al variare dell'orizzonte.

Avvio:  dashboard_whatif/.venv_dashboard/bin/streamlit run dashboard_whatif/app.py
Dati:   dashboard_whatif/dati/{breve,lungo}/ prodotti da genera_breve.py e genera_lungo.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATI = Path(__file__).resolve().parent/'dati'
BASELINE = {'Persistenza', 'Naive 24h', 'Naive 168h',
            'Naive settimanale', 'Profilo 8 settimane', 'Anno precedente'}
BLU, ARANCIO, GRIGIO, INCHIOSTRO = '#2f6db5', '#c7652b', '#b5b3ad', '#1d1d1b'
# Un naive è una previsione senza modello: ripete valori già osservati.
NAIVE = {
    'Persistenza': 'ripete l\'ultimo valore osservato',
    'Naive 24h': 'ripete il consumo della stessa ora del giorno prima',
    'Naive 168h': 'ripete il consumo della stessa ora della settimana prima',
    'Naive settimanale': 'ripete gli ultimi 7 giorni osservati per tutto l\'orizzonte',
    'Profilo 8 settimane': 'media per giorno della settimana sulle ultime 8 settimane',
    'Anno precedente': 'stesso giorno della settimana di 364 giorni prima, riscalato sul livello recente',
}

TAB = {
    'breve': dict(titolo='Breve termine · orario', passo='h', unita='kWh all\'ora',
                  slot={'1h': 1, '2h': 2, '6h': 6, '24h': 24, '48h': 48, '72h': 72, '96h': 96},
                  default='48h', storico=72, naive='Naive 24h', col_passo='ora', col_tempo='istante',
                  predefinito='Chronos-2'),
    'lungo': dict(titolo='Medio termine · giornaliero', passo='D', unita='kWh al giorno',
                  slot={'1 mese': 30, '2 mesi': 60, '3 mesi': 90},
                  default='3 mesi', storico=90, naive='Naive settimanale', col_passo='passo',
                  col_tempo='giorno', predefinito='Chronos-2'),
}

st.set_page_config(page_title='What-if previsioni U141', layout='wide')
st.markdown('<style>.block-container{padding-top:1.5rem} [data-testid=stMetricValue]{font-size:2.4rem}</style>',
            unsafe_allow_html=True)


@st.cache_data
def carica(tipo):
    cartella = DATI/tipo
    if not (cartella/'previsioni.csv').exists():
        return None
    p = pd.read_csv(cartella/'previsioni.csv')
    conf = json.loads((cartella/'configurazione.json').read_text())
    serie = pd.read_csv(cartella/'serie.csv', index_col=0, parse_dates=True).iloc[:, 0]
    if tipo == 'lungo':
        p['giorno'] = pd.to_datetime(p['giorno'])
        p['passo'] = (p['giorno']-pd.to_datetime(p['origine'])).dt.days+1
    else:
        p['istante'] = pd.to_datetime(p['istante'])
    p['ae'] = (p['punto']-p['reale']).abs()
    return p, conf, serie


@st.cache_data
def curve_mae(tipo):
    """MAE cumulato 1…k per ogni modello e ogni passo k (tutte le origini insieme)."""
    p, _, _ = carica(tipo)
    col = TAB[tipo]['col_passo']
    per_passo = p.groupby(['modello', col])['ae'].agg(['sum', 'count']).groupby(level=0).cumsum()
    return (per_passo['sum']/per_passo['count']).unstack(0)


@st.cache_data
def fascia_errori(tipo):
    """Quantili 10% e 90% dell'errore (reale − previsto) per modello e passo, su tutte le origini di test."""
    p, _, _ = carica(tipo)
    errore = (p['reale']-p['punto']).rename('errore')
    return errore.groupby([p['modello'], p[TAB[tipo]['col_passo']]]).quantile([.1, .9]).unstack()


def grafico_previsione(cfg, p, serie, modello, origine, h, fascia):
    sel = p[(p.modello == modello) & (p.origine == origine)].sort_values(cfg['col_passo'])
    t0 = sel[cfg['col_tempo']].iloc[0]
    passato = serie.loc[t0-pd.Timedelta(cfg['storico'], cfg['passo']):t0-pd.Timedelta(1, cfg['passo'])]
    dentro = sel[sel[cfg['col_passo']] <= h]
    fine = dentro[cfg['col_tempo']].iloc[-1]
    fig = go.Figure()
    fig.add_vrect(x0=t0, x1=fine, fillcolor=BLU, opacity=.06, line_width=0)
    del_modello = 'q10' in dentro and dentro['q10'].notna().any()
    if del_modello:      # intervallo prodotto dal modello
        basso, alto, nome = dentro['q10'], dentro['q90'], 'intervallo 10–90% del modello'
    else:                # fascia di errore empirica: previsione + quantili degli errori a quel passo
        q = fascia.loc[modello].reindex(dentro[cfg['col_passo']])
        basso = (dentro['punto']+q[.1].values).clip(lower=0)
        alto = (dentro['punto']+q[.9].values).clip(lower=0)
        nome = 'fascia di errore 10–90%'
    fig.add_trace(go.Scatter(x=dentro[cfg['col_tempo']], y=alto, line_width=0, showlegend=False,
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dentro[cfg['col_tempo']], y=basso, fill='tonexty', line_width=0,
                             fillcolor='rgba(47,109,181,.18)', name=nome, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=passato.index, y=passato.values, name='storico', line=dict(color=GRIGIO, width=1.5)))
    fig.add_trace(go.Scatter(x=sel[cfg['col_tempo']], y=sel['reale'], name='reale',
                             line=dict(color=INCHIOSTRO, width=1.5)))
    fig.add_trace(go.Scatter(x=dentro[cfg['col_tempo']], y=dentro['punto'], name=modello,
                             line=dict(color=BLU, width=3)))
    fig.add_vline(x=t0, line_dash='dot', line_color='#888')
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', y=1.08),
                      yaxis_title=cfg['unita'], hovermode='x unified', plot_bgcolor='#fafaf8')
    return fig, del_modello


def indicatore(mase):
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=mase, number=dict(valueformat='.2f'),
        gauge=dict(axis=dict(range=[0, 2]), bar=dict(color=INCHIOSTRO, thickness=.25),
                   steps=[dict(range=[0, .8], color='#cfe8d8'), dict(range=[.8, 1], color='#f3e3b5'),
                          dict(range=[1, 2], color='#f2c9c2')],
                   threshold=dict(line=dict(color=INCHIOSTRO, width=3), value=1))))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=0))
    return fig


def classifica(mae_h, modello, unita):
    s = mae_h.sort_values(ascending=True)
    colori = [BLU if m == modello else (GRIGIO if m in BASELINE else '#9dbbe0') for m in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h', marker_color=colori,
                           text=[f'{v:,.0f}' for v in s.values], textposition='outside'))
    fig.update_layout(height=34*len(s)+40, margin=dict(l=10, r=40, t=10, b=10), xaxis_title=f'MAE ({unita})',
                      yaxis=dict(autorange='reversed'), plot_bgcolor='#fafaf8')
    return fig


def crescita(curve, modello, naive, h, cfg):
    fig = go.Figure()
    for m, colore in [(naive, GRIGIO), (modello, BLU)]:
        fig.add_trace(go.Scatter(x=curve.index, y=curve[m], name=m, line=dict(color=colore, width=2.5)))
    fig.add_vline(x=h, line_dash='dot', line_color=INCHIOSTRO)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', y=1.15),
                      xaxis_title='orizzonte (' + ('ore' if cfg['passo'] == 'h' else 'giorni') + ')',
                      yaxis_title='MAE cumulato', plot_bgcolor='#fafaf8')
    return fig


def mostra(tipo):
    cfg = TAB[tipo]
    caricati = carica(tipo)
    if caricati is None:
        st.info(f'Dati non ancora generati: lancia `genera_{tipo}.py`.')
        return
    p, conf, serie = caricati
    curve = curve_mae(tipo)
    modelli = list(dict.fromkeys(p['modello']))

    c1, c2, c3 = st.columns([2.2, 1.3, 2])
    nome_slot = c1.segmented_control('Orizzonte', list(cfg['slot']), default=cfg['default'], key=f'slot_{tipo}')
    h = cfg['slot'][nome_slot or cfg['default']]
    naive_presenti = [m for m in modelli if m in NAIVE]
    modello = c2.selectbox('Modello', modelli, index=modelli.index(cfg['predefinito']), key=f'mod_{tipo}',
                           help='Modelli di riferimento senza addestramento (naive):\n\n'
                           + '\n'.join(f'- **{m}**: {NAIVE[m]}' for m in naive_presenti))
    if modello in NAIVE:
        c2.caption(f'{modello}: previsione senza modello, {NAIVE[modello]}.')
    origini = list(dict.fromkeys(p['origine']))
    origine = c3.select_slider('Origine della previsione', origini, value=origini[len(origini)//2],
                               key=f'orig_{tipo}')

    mae_h = curve.loc[h]
    mae, mase = mae_h[modello], mae_h[modello]/conf['scala_mase']
    sx, dx = st.columns([2.6, 1])
    fig, del_modello = grafico_previsione(cfg, p, serie, modello, origine, h, fascia_errori(tipo))
    sx.plotly_chart(fig, width='stretch')
    sx.caption('Fascia azzurra: intervallo 10–90% prodotto dal modello.' if del_modello else
               'Fascia azzurra: fascia di errore 10–90%, cioè dove è caduto l\'80% degli errori di questo modello '
               'alla stessa distanza dall\'origine, su tutte le origini di test. È una stima a posteriori, '
               'non un intervallo prodotto dal modello.')
    with dx:
        st.metric(f'Errore medio (MAE) a {nome_slot}', f'{mae:,.0f}', help=f'{cfg["unita"]}, media su tutte le '
                  f'origini di test, passi 1…{h}')
        st.caption(f'{cfg["unita"]} · posizione {list(mae_h.sort_values().index).index(modello)+1}'
                   f' su {len(mae_h)}')
        st.markdown(f'**Rispetto al {cfg["naive"]}** (MASE)')
        st.plotly_chart(indicatore(mase), width='stretch')
        st.caption('sotto 1 = meglio del naive, sopra 1 = peggio')
        st.caption(f'{cfg["naive"]}: previsione senza modello, {NAIVE[cfg["naive"]]}.')

    st.divider()
    a, b = st.columns([1.4, 1])
    a.markdown(f'**Classifica a {nome_slot}**')
    a.plotly_chart(classifica(mae_h, modello, cfg['unita']), width='stretch')
    b.markdown('**Come cresce l\'errore allontanandosi**')
    b.plotly_chart(crescita(curve, modello, cfg['naive'], h, cfg), width='stretch')


st.title('Previsione dei consumi U141: cosa succede se guardo più lontano?')
t1, t2 = st.tabs([TAB['breve']['titolo'], TAB['lungo']['titolo']])
with t1:
    mostra('breve')
with t2:
    mostra('lungo')

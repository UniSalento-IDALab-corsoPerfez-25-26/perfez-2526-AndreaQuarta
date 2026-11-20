"""Dati EWELD C24: un'unica fonte per tutti i modelli.

Ogni modello riceve gli stessi esempi (stesse origini, stesso storico, stessa
normalizzazione stimata solo sul train). Nessun training all'import.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parent.parent
CARTELLA_C24 = RADICE/'datasets/EWELD/Electricity Consumption/C/C24 Manufacture of basic metals'

# Capodanno lunare: periodo di fermo noto in anticipo (da -7 a +10 giorni).
CAPODANNO_LUNARE = ['2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05',
                    '2020-01-25', '2021-02-12', '2022-02-01']


def utenti_disponibili():
    return sorted(p.stem for p in CARTELLA_C24.glob('U*.csv'))


def carica_serie(utente):
    """Consumo orario in kWh: somma di quattro misure da 15 minuti, NaN se ne manca una."""
    path = CARTELLA_C24/f'{utente}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Utenza non trovata: {path}')
    d = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(d['Time']))
    if idx.has_duplicates:
        raise ValueError(f'Timestamp duplicati in {path.name}')
    s = pd.Series(pd.to_numeric(d['Value'], errors='raise').to_numpy(), index=idx).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)
    if (s.dropna() < 0).any():
        raise ValueError(f'Consumi negativi in {path.name}')
    serie = s.resample('h').sum(min_count=4).asfreq('h')
    serie.name = utente
    return serie


def calendario(index):
    """Variabili note in anticipo: ora, settimana, anno (seno/coseno), weekend, Capodanno lunare."""
    ora = index.hour.to_numpy()
    settimana = index.dayofweek.to_numpy()*24+ora
    giorno = index.dayofyear.to_numpy()
    festa = np.zeros(len(index))
    for data in CAPODANNO_LUNARE:
        d = pd.Timestamp(data)
        festa = np.maximum(festa, ((index >= d-pd.Timedelta(days=7)) & (index <= d+pd.Timedelta(days=10))))
    return np.column_stack([np.sin(2*np.pi*ora/24), np.cos(2*np.pi*ora/24),
                            np.sin(2*np.pi*settimana/168), np.cos(2*np.pi*settimana/168),
                            np.sin(2*np.pi*giorno/366), np.cos(2*np.pi*giorno/366),
                            index.dayofweek.to_numpy() >= 5, festa]).astype('float32')


N_CALENDARIO = calendario(pd.DatetimeIndex(['2020-01-01'])).shape[1]


@dataclass
class Finestre:
    """Esempi supervisionati: storico (n, lookback, 1+cal), calendario futuro (n, H, cal), target (n, H)."""
    storico: np.ndarray
    futuro: np.ndarray
    target_z: np.ndarray      # target standardizzato (per le loss)
    target: np.ndarray        # target in kWh (per le metriche)
    origini: pd.DatetimeIndex

    def __len__(self):
        return len(self.origini)


@dataclass
class Dati:
    utente: str
    serie: pd.Series
    periodi: dict
    horizon: int
    lookback: int
    ore_breve: int            # ore previste a ogni origine oraria (breve termine)
    media: float
    dev_std: float
    scala_mase: float         # MAE del naive 24h sul train
    train: Finestre
    validazione: Finestre
    test: Finestre            # origini ogni `horizon` ore: valutazione diretta e autoregressiva
    test_orario: Finestre     # origini ogni ora, target +1…+ore_breve: breve termine e one-step-ahead
    calendario_esteso: np.ndarray  # (n_test, 2·horizon−1, cal): calendario per la ricorsione
    scartate: dict = field(default_factory=dict)

    def a_kwh(self, z):
        return np.asarray(z)*self.dev_std+self.media

    def riepilogo(self):
        righe = []
        for nome in ('train', 'validazione', 'test', 'test_orario'):
            inizio, fine = self.periodi[nome.removesuffix('_orario')]
            f = getattr(self, nome)
            righe.append(dict(split=nome, inizio=inizio, fine=fine, origini=len(f),
                              scartate=self.scartate.get(nome, 0)))
        return pd.DataFrame(righe).set_index('split')


def _origini(inizio, fine, horizon, passo):
    """Origini con l'intero blocco di `horizon` ore dentro [inizio, fine]."""
    inizio = pd.Timestamp(inizio)
    ultima = pd.Timestamp(fine)+pd.Timedelta(days=1)-pd.Timedelta(hours=horizon)
    return pd.date_range(inizio, ultima, freq=f'{passo}h')


def _finestre(serie, origini, z, media, dev_std, horizon, lookback, primo_storico=None, ore_target=None):
    """ore_target: quante ore future servono come target (default tutto l'orizzonte)."""
    storici, futuri, target, valide = [], [], [], []
    scartate = 0
    for o in origini:
        idx_passato = pd.date_range(o-pd.Timedelta(hours=lookback), periods=lookback, freq='h')
        idx_futuro = pd.date_range(o, periods=horizon, freq='h')
        if primo_storico is not None and idx_passato[0] < primo_storico:
            scartate += 1
            continue
        passato, futuro = z.reindex(idx_passato), serie.reindex(idx_futuro[:ore_target or horizon])
        if passato.isna().any() or futuro.isna().any():
            scartate += 1
            continue
        storici.append(np.column_stack([passato.to_numpy(), calendario(idx_passato)]))
        futuri.append(calendario(idx_futuro))
        target.append(futuro.to_numpy())
        valide.append(o)
    if not valide:
        raise ValueError('Nessuna finestra completa: controllare periodi e dati mancanti.')
    target = np.asarray(target, dtype='float64')
    return Finestre(np.asarray(storici, dtype='float32'), np.asarray(futuri, dtype='float32'),
                    ((target-media)/dev_std).astype('float32'), target,
                    pd.DatetimeIndex(valide)), scartate


def prepara_dati(utente, periodi, horizon=48, lookback=168, passo_train=6, passo_validazione=24, ore_breve=6):
    """Divide la serie per date e costruisce le finestre di ogni split.

    periodi: {'train': (inizio, fine), 'validazione': (...), 'test': (...)}, date incluse.
    Le origini di test sono a blocchi non sovrapposti (passo = horizon): ogni ora
    di test viene valutata una sola volta. Lo storico all'origine contiene solo
    valori reali già osservati; il modello non viene riaddestrato sul test.
    `test_orario` ha invece un'origine per ogni ora di test, con target le `ore_breve` ore
    successive (breve termine); la one-step-ahead ne usa solo la prima.
    """
    serie = carica_serie(utente)
    p = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in periodi.items()}
    if not (p['train'][1] < p['validazione'][0] <= p['validazione'][1] < p['test'][0] <= p['test'][1]):
        raise ValueError('I periodi devono essere in ordine: train < validazione < test.')
    fine_train = p['train'][1]+pd.Timedelta(days=1)
    storico_train = serie.loc[p['train'][0]:fine_train-pd.Timedelta(hours=1)]
    if storico_train.notna().sum() < 30*24:
        raise ValueError('Train con meno di 30 giorni di dati validi.')

    media = float(storico_train.mean())
    dev_std = max(float(storico_train.std(ddof=0)), 1e-6)
    scala_mase = float(storico_train.diff(24).abs().mean())
    z = (serie-media)/dev_std

    passi = dict(train=passo_train, validazione=passo_validazione, test=horizon)
    finestre, scartate = {}, {}
    for nome, (inizio, fine) in p.items():
        # Nel train lo storico deve iniziare dentro il train stesso.
        if nome == 'train':
            inizio = inizio+pd.Timedelta(hours=lookback)
        origini = _origini(inizio, fine, horizon, passi[nome])
        finestre[nome], scartate[nome] = _finestre(serie, origini, z, media, dev_std, horizon, lookback,
                                                   primo_storico=p['train'][0])
    # Breve termine e one-step-ahead: un'origine per ogni ora di test, storico reale fino all'ora
    # precedente, target le `ore_breve` ore successive tutte dentro il test.
    if not 1 <= ore_breve <= horizon:
        raise ValueError(f'ore_breve deve essere tra 1 e horizon ({horizon}).')
    origini_orarie = _origini(*p['test'], horizon=ore_breve, passo=1)
    finestre['test_orario'], scartate['test_orario'] = _finestre(
        serie, origini_orarie, z, media, dev_std, horizon, lookback, primo_storico=p['train'][0],
        ore_target=ore_breve)
    # Ricorsione: alla k-esima ora prevista serve il calendario delle `horizon` ore successive.
    calendario_esteso = np.stack([calendario(pd.date_range(o, periods=2*horizon-1, freq='h'))
                                  for o in finestre['test'].origini])
    date ={k: (str(a.date()), str(b.date())) for k, (a, b) in p.items()}
    return Dati(utente, serie, date, horizon, lookback, ore_breve, media, dev_std, scala_mase,
                finestre['train'], finestre['validazione'], finestre['test'], finestre['test_orario'],
                calendario_esteso, scartate)


def contesti(dati, origini, ore):
    """Storico lungo in kWh per i modelli preaddestrati: le `ore` precedenti a ogni origine.

    Stessa serie e stesse origini degli altri modelli; cambia solo la lunghezza dello storico.
    Eventuali buchi vengono interpolati usando solo valori passati all'origine.
    """
    righe = []
    for o in origini:
        idx = pd.date_range(o-pd.Timedelta(hours=ore), periods=ore, freq='h')
        if idx[0] < pd.Timestamp(dati.periodi['train'][0]):
            raise ValueError(f'Contesto di {ore}h prima dell’inizio del train per l’origine {o}.')
        s = dati.serie.reindex(idx)
        if s.isna().all():
            raise ValueError(f'Contesto senza dati per l’origine {o}.')
        righe.append(s.interpolate(limit_direction='both').to_numpy('float32'))
    return np.stack(righe)

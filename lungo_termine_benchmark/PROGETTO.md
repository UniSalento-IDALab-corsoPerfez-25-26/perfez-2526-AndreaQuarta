# Benchmark a lungo termine EWELD C24 — documento di progetto

Stato: bozza da approvare · 2026-10-24

## 1. Perché

`c24_benchmark/` valuta previsioni orarie a 48h (breve/medio termine) e resta com'è.
Serve un secondo benchmark, costruito sulla stessa falsariga, per rispondere a:
**quanto bene si prevede il consumo giornaliero di un'utenza C24 da 30 a 90 giorni avanti
(mese e trimestre successivo), e quale famiglia di modelli lo fa meglio?**

## 2. Obiettivi, non-obiettivi, anti-obiettivi

**Obiettivi**
- Previsione del consumo **giornaliero** (kWh/giorno) con orizzonte **H = 90 giorni**,
  valutata anche ai tagli 30 e 60 giorni.
- Valutazione sia giorno per giorno sia sul **totale del periodo** (mese, trimestre).
- Stesso protocollo per tutti i modelli: stessi dati, stesse origini, stesse metriche.
- U141 come utenza principale (confrontabile con `c24_benchmark`), più un confronto
  multi-utente.

**Non-obiettivi**
- Nessuna previsione oraria a lungo termine (né disaggregazione oraria).
- Nessun meteo: a 30–90 giorni non esiste una previsione affidabile.
- Nessuna modifica a `c24_benchmark/`.

**Anti-obiettivi** (il benchmark sarebbe un fallimento anche se "gira")
- Modelli che imparano dai tratti piatti/zeri finti del dataset (vedi §3).
- Data leakage: valori futuri all'origine usati come ingresso, o modelli globali addestrati
  su dati di altre utenze successivi alla data di taglio.
- Un test che non contiene un fermo del Capodanno lunare: è l'evento annuale che conta.

## 3. Cosa dicono i dati (verificato il 2026-10-24)

- 23 utenze in `datasets/EWELD/Electricity Consumption/C/C24 Manufacture of basic metals`,
  misure ogni 15 minuti, nominalmente complete al 100%.
- **I buchi sono riempiti con valori costanti** (quasi sempre 0, a volte un valore fisso
  come 63.5): tratti piatti fino a 840 giorni. Vanno trattati come dati mancanti.
- **Fermo annuale del Capodanno lunare**: ogni anno (2018, 2019, 2020) tra gennaio e
  febbraio il consumo crolla del 50–95% per 4–6 settimane, poi riprende. Non è piatto,
  quindi la pulizia non lo tocca. La finestra di `c24_benchmark` (−7/+10 giorni) è troppo
  stretta per questo fermo.
- Il 2020 **non** mostra un cambio di regime rispetto agli anni precedenti (il fermo di
  inizio 2020 è in linea con gli altri), quindi resta nel test.

## 4. Dati e pulizia (`dati.py`)

1. Caricamento come in `c24_benchmark/dati.py::carica_serie` (somma oraria di 4 misure).
2. Ogni tratto **costante per ≥ 24 ore consecutive** (qualsiasi valore) diventa NaN.
3. Aggregazione giornaliera: somma delle 24 ore, **NaN se manca anche una sola ora**.
4. Nessuna interpolazione nel target. Per i modelli che non accettano NaN in ingresso, lo
   storico si interpola usando solo valori passati all'origine (come `contesti` oggi).
5. Normalizzazione (media/dev. std) stimata solo sul train, come in `c24_benchmark`.

## 5. Protocollo di valutazione

**Split comune per tutte le utenze** (date incluse):

| Parte | Periodo | Uso |
|---|---|---|
| train | inizio dati → 2018-09-30 | addestramento |
| validazione | origini mensili 2018-10-01 … 2019-01-01 (finestre fino al 2019-03-31) | early stopping, scelta iperparametri; contiene il fermo 2019 |
| test | origini il 1° di ogni mese, 2019-04-01 … 2020-05-01 (14 origini) | metriche; contiene il fermo 2020 |

- Dopo la validazione, ogni modello viene **riaddestrato su train + validazione**
  (dati fino al 2019-03-31) con gli iperparametri scelti, poi **non** più riaddestrato
  sul test (come `c24_benchmark`).
- A ogni origine il modello riceve **tutto lo storico reale precedente** (giornaliero) e il
  calendario dei 90 giorni futuri; nessun consumo futuro.
- Le finestre di test **si sovrappongono** (origini mensili, H = 90): ogni giorno viene
  previsto più volte a orizzonti diversi. È voluto: diversamente da `c24_benchmark`
  interessa l'errore in funzione della distanza.
- Un'origine si valuta solo se i 90 giorni target sono tutti validi; le origini scartate
  vanno riportate per utenza.
- Un solo protocollo di previsione: **diretto** (90 valori in un colpo o, per le baseline,
  regole che non usano valori futuri). Niente one-step né ricorsione.

**Utenze**
- Principale: **U141** (14/14 origini di test valide, 545 giorni validi prima del test).
- Multi-utente: utenze con ≥ 365 giorni validi prima del 2019-04-01 e ≥ 6 origini di test
  valide. Oggi (ricalcolato su tutte le 23): **U135, U137, U138, U141, U142, U143, U144, U147,
  U148, U151, U155** (11). Escluse tra le altre U152 e U150 (311 giorni di train), U134 e
  U153 (5 origini valide), U140 e U136 (0 e 2 origini valide).
  La lista si ricalcola dal codice, non si scrive a mano.
- Sulle utenze diverse da U141 si usano gli iperparametri scelti su U141 (nessuna
  validazione per utenza).

## 6. Modelli (`modelli.py`)

| Famiglia | Modelli | Note |
|---|---|---|
| Baseline | Naive settimanale; Profilo 8 settimane; Anno precedente | vedi sotto |
| Statistici | STL settimanale + ETS sul destagionalizzato; ETS; SARIMA(1,0,1)(0,1,1)₇; regressione con Fourier annuale + giorni dal Capodanno lunare | `statsmodels`, già in `.venv_eweld` |
| Alberi | LightGBM, XGBoost | diretti, vedi sotto |
| Preaddestrati | Chronos-2, Chronos-Bolt, TimesFM 2.5, TimesFM 3.0 | zero-shot su serie giornaliera, venv e pesi già presenti |
| Reti per orizzonti lunghi | N-HiTS, PatchTST | **globali**: un modello addestrato su tutte le 11 utenze |

- **Naive settimanale**: ultimi 7 giorni osservati ripetuti per 90 giorni.
- **Profilo 8 settimane**: media per giorno della settimana sulle ultime 8 settimane.
- **Anno precedente**: valore dello stesso giorno della settimana 52 settimane prima
  (364 giorni), riscalato per il rapporto tra le ultime 4 settimane e le stesse 4 settimane
  dell'anno prima. Se mancano dati dell'anno prima, ripiega sul profilo 8 settimane.
- **Alberi diretti**: un esempio per (origine, giorno target). Feature solo note
  all'origine: orizzonte (1–90), calendario del giorno target (giorno della settimana,
  giorno dell'anno seno/coseno, festività), **giorni dal Capodanno lunare** del target
  (numerico, con segno), livello recente (medie ultime 7/28 giorni), valore dell'anno prima
  (NaN se assente). Target normalizzato per il livello recente. Origini di train
  giornaliere, con i 90 giorni target interamente dentro il periodo di addestramento.
- **SARIMA**: il modello statistico più usato nei lavori sui consumi industriali, qui come
  riferimento per la letteratura. Ordini fissi (nessuna selezione per origine), stimato a ogni
  origine come ETS; se la stima fallisce ripiega sul profilo 8 settimane.
- **Stagionalità annuale negli statistici**: MSTL con periodo 365 richiede ≥ 2 anni di
  train e U141 ne ha circa 1,5, quindi l'annuale entra come regressori (Fourier + giorni
  dal Capodanno lunare) invece che come componente STL.
- **Preaddestrati**: contesto = tutto lo storico giornaliero disponibile (max 2048 giorni),
  previsione a 90 passi; si usa la mediana come previsione puntuale e i quantili 10/90
  per la copertura.
- **N-HiTS / PatchTST**: addestrati solo su dati ≤ 2018-09-30 (tuning) e poi ≤ 2019-03-31
  per tutte le utenze, così nessun dato di altre utenze sovrapposto al test entra
  nell'addestramento.

## 7. Metriche (`metriche.py` nella radice, unico per tutti i benchmark)

Calcolate in kWh/giorno su tutte le origini di test valide, per ogni taglio di orizzonte
(1–30, 1–60, 1–90 giorni):

- MAE, RMSE, sMAPE %, Bias (come in `c24_benchmark`);
- **MAPE %** sui giorni con consumo reale non nullo, solo per il confronto con la letteratura:
  nei giorni di fermo il consumo è vicino a zero e il MAPE esplode (su U141 è 150–375%),
  quindi non si usa per scegliere i modelli;
- **MASE** con scala = MAE del naive settimanale (lag 7) sul train;
- **errore % sul totale** del periodo: |Σ previsto − Σ reale| / Σ reale, per mese (1–30)
  e trimestre (1–90);
- **copertura dell'intervallo 80%** e larghezza media, solo per i modelli che producono
  quantili;
- errore per orizzonte (curva MAE vs giorno 1–90) e, separatamente, l'errore sulle
  finestre che contengono il fermo 2020;
- **test di Diebold-Mariano** di ogni modello contro il naive settimanale, per utente e per
  taglio (`dm.csv`): perdita = MAE di ogni origine, correzione di Harvey-Leybourne-Newbold,
  t di Student a n−1 gradi. Le origini sono mensili e le finestre si sovrappongono, quindi le
  perdite sono autocorrelate fino al ritardo h−1 con h = ⌈taglio/30⌉ (1, 2, 3). Con 14 origini
  il test ha poca potenza: un p alto non vuol dire che i modelli si equivalgono.

## 8. Struttura e output

```
lungo_termine_benchmark/
  PROGETTO.md        questo documento
  dati.py            caricamento, pulizia, aggregazione, split, finestre
  modelli.py         crea_modello(nome), stessa interfaccia addestra/prevedi_test
  principale.ipynb   U141 + multi-utente, tabelle e grafici
  risultati/<utente o multi>_<timestamp>/  configurazione.json, metriche_*.csv, previsioni.csv, dm.csv
```

Stile del codice, lingua (italiano) e organizzazione come `c24_benchmark`. I modelli
preaddestrati girano nei rispettivi venv tramite sottoprocesso, come lì.

## 9. Scelte fatte da Claude (da controllare)

- Soglia della pulizia: ≥ 24 ore costanti → NaN; giorno NaN se manca un'ora.
- Split comune con le date del §5 e riaddestramento su train + validazione.
- Origini mensili sovrapposte e un solo protocollo diretto.
- Criteri di inclusione multi-utente (≥ 365 giorni di train, ≥ 6 origini valide).
- Iperparametri di U141 riusati sulle altre utenze.
- Meteo escluso; 2020 incluso nel test.
- N-HiTS/PatchTST con `neuralforecast` in un **nuovo venv `.venv_neural`**, per non
  toccare le versioni di `.venv_eweld` (è l'unica nuova dipendenza).

## 10. Lasciato a chi implementa

- Iperparametri iniziali e griglie di ricerca.
- Dettagli dell'interfaccia interna, purché `principale.ipynb` resti leggibile come
  quello di `c24_benchmark`.
- Grafici del notebook.

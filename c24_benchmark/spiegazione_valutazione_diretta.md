# Come funziona la valutazione diretta 48h (sezione 3.1)

Esempio con i numeri reali di **XGBoost sull'utente U141**: sono le previsioni che compaiono
nella tabella 3.1 di `principale.ipynb`.

Configurazione: train 2018-11-01 → 2019-09-30, validazione 2019-10-01 → 2019-10-31,
test 2019-11-01 → 2019-12-31.

---

## Passo 1: si divide il test in blocchi da 48 ore

Il test va dal 1° novembre al 31 dicembre 2019, cioè 61 giorni. Si divide in **30 blocchi
da 48 ore**, uno per origine:

```
origine 1:  1 nov 00:00 → prevede 1 nov 00:00 … 2 nov 23:00
origine 2:  3 nov 00:00 → prevede 3 nov 00:00 … 4 nov 23:00
origine 3:  5 nov 00:00 → ...
...
origine 30: 29 dic 00:00 → prevede 29 dic … 30 dic
```

I blocchi non si sovrappongono, quindi ogni ora di test viene prevista e valutata una sola volta.

## Passo 2: cosa riceve il modello all'origine 1

Facciamo finta che siano le **00:00 del 1° novembre**. Il modello riceve:

- **le ultime 168 ore di consumo reale**, dal 25 ottobre 00:00 al 31 ottobre 23:00. Le ultime quattro sono:
  ```
  31 ott 20:00 → 7240 kWh
  31 ott 21:00 → 6965 kWh
  31 ott 22:00 → 7108 kWh
  31 ott 23:00 → 6910 kWh
  ```
- **il calendario delle 48 ore future**: ora del giorno, giorno della settimana, weekend,
  Capodanno lunare. Sono informazioni note in anticipo.

**Non riceve** nessun consumo del 1° o del 2 novembre: alle 00:00 del 1° novembre quei valori
non esistono ancora. Usarli sarebbe *data leakage*.

Nel codice (`dati.py`, funzione `_finestre`):
- `storico` contiene solo le ore **prima** dell'origine;
- `futuro` contiene solo il **calendario** delle 48 ore, nessun consumo;
- `target` contiene i consumi reali delle 48 ore, usati **solo** per calcolare le metriche e
  mai come ingresso del modello.

## Passo 3: il modello restituisce 48 numeri in un colpo

Alcune delle 48 previsioni dell'origine 1, confrontate con i consumi reali:

| Ora | Previsto | Reale | Errore | \|Errore\| |
|---|---|---|---|---|
| +1h (1 nov 00:00) | 6226 | 6229 | −4 | 4 |
| +2h (1 nov 01:00) | 6129 | 6427 | −298 | 298 |
| +3h (1 nov 02:00) | 6057 | 5977 | +81 | 81 |
| +12h (1 nov 11:00) | 5630 | 4032 | **+1598** | 1598 |
| +24h (1 nov 23:00) | 6307 | 7152 | −845 | 845 |
| +25h (2 nov 00:00) | 6302 | 6932 | −630 | 630 |
| +48h (2 nov 23:00) | 6375 | 7262 | −887 | 887 |

Anche la previsione delle 23:00 del 2 novembre (+48h) è fatta **alle 00:00 del 1° novembre**,
senza sapere niente di quello che succede nel frattempo. Per questo è una previsione a due
giorni.

La media dei 48 errori assoluti di questa sola origine è **MAE = 764.6 kWh**.

## Passo 4: si passa all'origine successiva

Sono le **00:00 del 3 novembre**. Le 48 ore precedenti sono passate, quindi i consumi reali del
1° e del 2 novembre **entrano nello storico**. Il modello è sempre lo stesso: non viene
riaddestrato, riceve solo dati più aggiornati, come succederebbe in produzione.

| Ora | Previsto | Reale | Errore |
|---|---|---|---|
| +1h (3 nov 00:00) | 6871 | 7471 | −600 |
| +12h (3 nov 11:00) | 5561 | 4878 | +683 |
| +48h (4 nov 23:00) | 6444 | 7855 | −1412 |

MAE di questa origine: **693.2 kWh**. Si ripete così per tutte le 30 origini.

## Passo 5: come nasce la metrica della tabella

Alla fine si ha una matrice di **30 origini × 48 ore = 1440 previsioni** e altrettanti valori
reali. `metriche.calcola` calcola i 1440 errori (previsto − reale) e da questi:

| Metrica | Calcolo | Risultato |
|---|---|---|
| **MAE** | media dei 1440 \|errori\| | **655.9** |
| **RMSE** | √(media dei 1440 errori²) | 833.0 |
| **Bias** | media dei 1440 errori con segno | −127.2: in media sottostima |
| **MASE** | MAE / 693.6, l'errore tipico del naive 24h sul train | 655.9 / 693.6 = **0.946** |

Sono esattamente i numeri della riga XGBoost nella tabella 3.1.

## Cosa significa in pratica

**MAE 655.9** significa: *"se ogni due giorni, a mezzanotte, chiedo a XGBoost il consumo delle
48 ore successive, sbaglia in media di circa 656 kWh per ora."*

È una media che mescola ore facili e ore difficili:
- primo giorno (+1h … +24h): MAE **635**
- secondo giorno (+25h … +48h): MAE **676**, un po' peggio perché è più lontano

Anche fra le origini la variabilità è grande: la più facile ha MAE 339, la più difficile 1373.
Per questo conviene usare 30 origini invece di una sola.

---

## Confronto con le altre tre valutazioni

| Valutazione | Origini | Cosa vede il modello | Ore valutate |
|---|---|---|---|
| **Diretta 48h** (3.1) | ogni 48 ore | storico reale fino all'origine | +1 … +48 |
| **Breve termine 6h** (3.2) | ogni ora | storico reale fino all'ora precedente | +1 … +6 |
| **One-step-ahead** (3.3) | ogni ora | storico reale fino all'ora precedente | solo +1 |
| **Autoregressiva 48h** (3.4) | ogni 48 ore | storico reale fino all'origine, poi le **proprie** previsioni reinserite ora per ora | +1 … +48 |

Il modello addestrato è lo stesso in tutti e quattro i casi: cambia solo il modo di usarlo sul test.

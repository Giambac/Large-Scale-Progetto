# Designing Large Scale AIS — GSD

## Phase 1

### Pacchetti

1. `hdbscan` → clustering.py
2. `numpy` → tutti i src/
3. `sentence-transformers` → embedding_store.py
4. `anthropic` → cluster_naming.py
5. `datasets` → data_loader.py
6. `google-generativeai` → cluster_naming.py
7. `hashlib`, `json`, `os` → data_loader.py
8. `dataclasses`, `enum` → state.py, stopping.py

---

### 5 wave

#### Wave 0 — "prepara i test prima di scrivere il codice"

Il GSD ha installato pytest e scritto 41 test che descrivono cosa il codice dovrà fare — ma il codice non esiste ancora, quindi tutti i test falliscono intenzionalmente. È come scrivere le domande dell'esame prima di studiare: sai esattamente cosa devi dimostrare.

#### Wave 1 — "scarica i dati e blindali"

Ha scaricato 15.000 recensioni Amazon (categoria Arts & Crafts) da HuggingFace. Le ha divise in due parti: 12.000 per addestrare, 3.000 da non toccare mai (il "held-out"). Ha calcolato un'impronta digitale SHA-256 del file held-out e l'ha salvata.

SHA-256 è un algoritmo che legge un file e produce una stringa di 64 caratteri. Questa stringa è come un'impronta digitale del file: se cambia anche solo una virgola nel contenuto, la stringa cambia completamente. Il GSD ha calcolato questa stringa subito dopo aver creato il file `held_out.jsonl` e l'ha salvata in un file separato `held_out.sha256`. Ogni volta che il sistema si avvia, ricalcola l'impronta del file e la confronta con quella salvata. Se non coincidono — qualcuno ha modificato il file, lo ha rigenerato, o ha fatto una copia sbagliata — il programma crasha con un errore esplicito invece di continuare silenziosamente con dati sbagliati.

Da questo momento, ogni volta che il sistema parte, controlla che quel file non sia stato modificato — se qualcuno lo tocca, il programma crasha immediatamente. Questo protegge l'integrità degli esperimenti futuri. Ha anche scritto il codice per i tre criteri di stop (turni esauriti, oracle soddisfatto, feedback che non cambia più) come strutture Python precise, non come commenti.

**Problema incontrato:** la libreria `datasets` versione 4.8.5 era rotta per questo dataset specifico → ha fatto il downgrade alla 2.21.0.

#### Wave 2 — "trasforma il testo in numeri"

Ha preso le 12.000 recensioni e le ha convertite in vettori numerici (embeddings) usando il modello `all-mpnet-base-v2`. Ogni recensione diventa un vettore di 768 numeri che cattura il suo significato. Il risultato è salvato in un file `.npy` da 36MB che non viene mai ricalcolato — è la base permanente su cui girano tutti i clustering.

#### Wave 3 — "raggruppa e dai nomi ai gruppi"

Ha scritto il codice per eseguire HDBSCAN sugli embeddings: un algoritmo che trova automaticamente quanti cluster esistono nei dati (senza doverlo specificare a priori). Per ogni cluster ha chiamato l'API di Claude per generare un nome e una descrizione. Ha anche costruito la struttura dati centrale `ClusteringState` — un oggetto Python che contiene l'assegnazione di ogni recensione al suo cluster, con anche una probabilità "morbida" (quanto è sicuro che quella recensione appartenga a quel cluster).

**Problemi incontrati e risolti:** HDBSCAN restituisce probabilità non normalizzate → aggiunta normalizzazione. Con dataset piccoli (usati nei test) il parametro `min_cluster_size=50` era troppo alto → auto-scaling automatico.

#### Wave 4 — "salva tutto e collega il pipeline"

Ha completato la serializzazione: ogni `ClusteringState` può essere scritto su file JSONL (un JSON per riga) e riletto identicamente. Ha scritto uno script `setup_phase1.py` che esegue l'intera catena end-to-end: verifica hash → carica embeddings → carica testi → clustering → nomi LLM → salva sul file di log.

---

### Embeddings

Il file contiene una tabella di 12.000 righe × 768 colonne. Ogni riga è una recensione, ogni colonna è un numero.

Quei 768 numeri non hanno un significato umano leggibile — non c'è una colonna che significa "positività" e una che significa "qualità del prodotto". Sono la rappresentazione interna del modello `all-mpnet-base-v2`, addestrato su miliardi di frasi. La proprietà chiave è questa: due recensioni che parlano della stessa cosa avranno vettori numericamente simili, due recensioni molto diverse avranno vettori lontani.

È su queste distanze che HDBSCAN lavora per trovare i cluster — non legge il testo, vede solo quanto sono vicini o lontani questi vettori nello spazio a 768 dimensioni.

---

### src

#### state.py

Definisce solo due "contenitori" di dati.

- `Cluster` rappresenta un singolo gruppo. Ha quattro campi: un numero identificativo, un nome generato dall'LLM, una descrizione generata dall'LLM, e la lista degli ID delle recensioni che appartengono a quel gruppo.
- `ClusteringState` rappresenta la fotografia completa del sistema in un dato momento della conversazione. Ha cinque campi: il numero del turno (0, 1, 2...), il timestamp, la lista dei cluster, un dizionario che dice per ogni recensione a quale cluster appartiene, e un dizionario che dice per ogni recensione quanto è "sicura" l'appartenenza a ciascun cluster.

```python
turn_index: 0
timestamp: "2026-05-05T10:00:00"
clusters: [
    Cluster(id=0, name="Positive Reviews", description="Happy customers",
    item_ids=[0, 2, 4]),
    Cluster(id=1, name="Negative Reviews", description="Unhappy customers",
    item_ids=[1, 3])
]
assignments: {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
soft_probs: {0: [0.9, 0.1], 1: [0.1, 0.9], 2: [0.8, 0.2], ...}
```

Il `soft_probs` della recensione 0 è `[0.9, 0.1]` — significa che il sistema è 90% sicuro che appartenga al cluster 0 e 10% al cluster 1. Una recensione con `[0.5, 0.5]` sarebbe invece un caso borderline ambiguo.

---

#### stopping.py

Quando si ferma la conversazione.

1. **Oracle soddisfatto** — l'utente dice esplicitamente "va bene così". È la condizione prioritaria, viene controllata per prima.
2. **Turni esauriti** — se si arriva al turno 15 senza che nessuno sia soddisfatto, il sistema si ferma comunque. È un tetto rigido per evitare loop infiniti.
3. **Feedback che non cambia più** — se l'utente continua a fare feedback sempre più piccoli ("sposta questa recensione", "cambia questo nome") il sistema capisce che sta convergendo e si ferma. Questa condizione è uno stub — la struttura c'è ma i valori numerici (quanto piccolo deve essere il feedback? per quanti turni?) vengono decisi nella Phase 4.

La funzione `check_stopping` le controlla nell'ordine 1 → 2 → 3 e restituisce il motivo dello stop, oppure `None` se la conversazione deve continuare.

`FeedbackMagnitudeWeights` è la struttura che in Phase 4 dirà quanto "pesa" ogni tipo di feedback — un feedback globale ("troppi cluster") pesa più di uno puntuale ("sposta questa recensione"). Per ora tutti i valori sono `nan`, segnaposto.

---

#### data_loader.py

1. **Scaricare i dati (una volta sola)**

   La funzione `download_and_save_dataset` va su HuggingFace, scarica in streaming le prime 15.000 recensioni di Arts & Crafts (senza scaricare tutti i 9 milioni di righe del dataset completo), e le salva in un file JSONL. Ogni riga del file è una recensione con il suo ID e il suo testo:

   ```python
   {"item_id": 0, "text": "Great product, very happy..."}
   {"item_id": 1, "text": "Broke after two uses..."}
   ```

   Se `held_out.jsonl` esiste già, la funzione crasha — non sovrascrive mai.

2. **Dividere i dati**

   La funzione `split_dataset` prende le 15.000 recensioni e le divide in modo riproducibile: 12.000 per il training (usate per clustering e conversazione) e 3.000 held-out (chiuse in cassaforte per la valutazione finale). Il `seed=42` garantisce che la divisione sia sempre identica — se la rifai domani ottieni gli stessi due gruppi.

3. **Proteggere il held-out**

   `compute_sha256` legge il file held-out e produce la sua impronta digitale. `verify_held_out_hash` la confronta con quella salvata e crasha se non coincidono. Questa funzione viene chiamata ogni volta che il sistema parte — è la guardia che impedisce contaminazioni accidentali.

---

#### embedding_store.py

È un contenitore semplice con una regola sola: carica una volta, non modificare mai.

La classe `EmbeddingStore` tiene in memoria la tabella 12.000×768. Ha tre metodi utili:

- `load(path)` — legge il file `.npy` dal disco e lo carica in memoria. È quello che viene chiamato ad ogni avvio del sistema.
- `get(item_id)` — restituisce il vettore di 768 numeri per una singola recensione. `get(0)` ti dà la prima recensione, `get(5)` la sesta, ecc.
- `get_all()` — restituisce tutta la tabella. È quello che HDBSCAN usa per trovare i cluster.

Il dettaglio importante è `flags.writeable = False` — dopo aver caricato la tabella, il codice la marca esplicitamente come sola lettura. Se qualcuno prova a modificare un valore crasha subito. Questo garantisce che gli embeddings rimangano identici per tutto il progetto — se cambiassero i numeri cambierebbero i cluster e i risultati degli esperimenti non sarebbero più confrontabili.

`compute_and_save` è il metodo che produce il file `.npy`.

`load_texts_from_jsonl` è una funzione di supporto che legge il file `train.jsonl` e restituisce la lista dei testi — serve come input a `compute_and_save` nella fase di setup.

---

#### clustering.py

Prende gli embeddings e produce il primo `ClusteringState`. Ha tre funzioni che si chiamano in sequenza.

1. **run_hdbscan** — prende la tabella 12.000×768 e trova i cluster automaticamente. Restituisce due cose: `labels` (un numero per ogni recensione che dice a quale cluster appartiene, oppure -1 se è "rumore") e `soft_probs` (la matrice delle probabilità morbide). Le righe di `soft_probs` vengono normalizzate perché HDBSCAN le restituisce non normalizzate — senza questa correzione non sommerebbero a 1.0.

2. **assign_noise_to_nearest** — risolve il problema dei punti rumore. HDBSCAN assegna -1 alle recensioni che non appartengono chiaramente a nessun cluster. Questa funzione le forza in un cluster guardando quale colonna di `soft_probs` ha il valore più alto per quella recensione — in pratica "al cluster a cui assomiglia di più". Alla fine ogni recensione ha un cluster, nessuno rimane -1.

3. **build_initial_clustering_state** — è la funzione principale che mette tutto insieme. Chiama `run_hdbscan`, poi `assign_noise_to_nearest`, poi raggruppa le recensioni per cluster, poi chiama l'LLM per dare un nome e una descrizione a ciascun cluster, e infine assembla il `ClusteringState` al turno 0. È il punto di ingresso del sistema — da qui in poi inizia la conversazione con l'oracle.

---

#### cluster_naming.py

Chiede all'LLM di dare un nome e una descrizione a ogni cluster.

Per ogni cluster prende le prime 5 recensioni e manda questo prompt all'LLM: *"Hai davanti 5 recensioni di un negozio di arts & crafts. Rispondi SOLO con un JSON con due chiavi: 'name' (2-5 parole) e 'description' (1-2 frasi)."*

L'LLM risponde tipo:

```python
{"name": "Knitting Supplies",
"description": "Reviews about yarn, needles and knitting accessories."}
```

Il codice controlla che la risposta abbia esattamente quelle due chiavi — se l'LLM risponde in modo sbagliato, crasha.

`ClusterNamer` è un'interfaccia. Dice solo "qualsiasi oggetto che ha un metodo `name_cluster` va bene". Questo permette di avere due implementazioni intercambiabili senza cambiare il resto del codice:

- `AnthropicClusterNamer` — usa Claude Haiku tramite API Anthropic
- `GoogleClusterNamer` — usa Gemini tramite Google AI Studio

`clustering.py` non sa quale dei due sta usando — riceve solo un oggetto che sa chiamare `name_cluster`. Puoi cambiare provider senza toccare nulla altro.

---

#### serialization.py

Risolve un problema pratico: come salvi un `ClusteringState` su file e lo rileggi identico.

**Il problema dei tipi**

Python e JSON non parlano la stessa lingua su due cose:

1. Numpy usa `float32` per i numeri — JSON non lo conosce e crasha. Il `_StateEncoder` li converte in `float` Python normale prima di scrivere.
2. JSON converte sempre le chiavi dei dizionari in stringhe. Quindi `{0: "cluster_0"}` diventa `{"0": "cluster_0"}` sul file. Quando rileggi, se fai `state.assignments[0]` ottieni `KeyError` perché la chiave è diventata la stringa `"0"`. `deserialize_state` risolve questo con `int(k)` su ogni chiave.

**Le quattro funzioni**

1. `serialize_state` — prende un `ClusteringState` e lo trasforma in una singola riga JSON senza newline.
2. `deserialize_state` — prende quella riga e ricostruisce il `ClusteringState` identico, incluso il cast delle chiavi a `int`.
3. `append_to_audit_log` — aggiunge una riga al file di log. Ogni turno della conversazione chiama questa funzione — il file cresce di una riga per turno e non si sovrascrive mai.
4. `load_audit_log` — rilegge l'intero file e restituisce la lista di tutti i `ClusteringState` di tutti i turni. Serve alle fasi successive per analizzare cosa è successo durante la conversazione.

---

### Script

#### setup_phase1.py

Chiama i 7 moduli in ordine, con stampe di avanzamento.

1. Verifica l'hash del held-out — se qualcuno l'ha toccato, crasha qui.
2. Carica `embeddings.npy` in un `EmbeddingStore`.
3. Carica i record da `train.jsonl` e controlla che il numero coincida con gli embeddings.
4. Esegue HDBSCAN + naming LLM e produce il `ClusteringState` al turno 0.
5. Salva quel `ClusteringState` come prima riga di `audit_log.jsonl`.

Ha anche tre flag utili per i test:

- `--n-items 500` — usa solo 500 recensioni invece di 12.000, molto più veloce
- `--min-cluster-size 20` — forza un parametro HDBSCAN specifico
- `--model gemini-1.5-flash` — usa Gemini invece di Claude per il naming

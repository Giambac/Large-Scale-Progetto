# Designing Large Scale AIS – Phase 2

## Pacchetti
1. `flask` $\rightarrow$ `web/app.py` – server web
2. `flask-socketio`$\rightarrow$ `web/app.py` – WebSocket per aggiornamenti live al browser
3. ` scikit-learn` $\rightarrow$ `src/clustering.py` – KMeans e GaussianMixture (BIC per selezione K)
4. `umap-learn` $\rightarrow$ `web/app.py` – proiezione 2D degli embeddings (_compute_projection)
5. `hdbscan` $\rightarrow$ già in fase 1, ma nella fase 2 è diventato lazy import (non più a livello di modulo)

### Libreria standard Python aggiunta
1. `argparse` $\rightarrow$ `web/app.py` – flag --backend
2. `datetime` $\rightarrow$ `web/app.py` – timestamp sessioni
3. `shutil` $\rightarrow$ `web/app.py` – copia embeddings.npy nella cartella sessione

## Wave
### Wave 0 – Scaffold dei test
Lo scopo della wave 0 è scrivere tutti i test prima che esista qualcosa da testare – è il principio TDD (Test-Driven Development): prima definisci cosa il sistema deve fare, poi lo costruisci.

**Tre nuove fixture in `tests/conftest.py` :**
1. `tiny_state_3cluster` è la fixture più importante di tutta la fase 2. È un `ClusteringState` artificiale con 3 cluster e 6 item, con le probabilità calibrate in modo che alcune recensioni siano chiaramente appartenenti a un cluster (item 4 ha `[0.05, 0.05, 0.90]` — quasi certamente nel cluster 2) e altre siano ambigue (item 1 ha `[0.70, 0.20, 0.10]` — meno sicuro). Questa ambiguità è deliberata: serve a verificare che il sistema riconosca correttamente i casi borderline.
2. `mock_embeddings_3cluster` produce 6 vettori da 768 numeri con seed fisso. Il seed garantisce che ogni volta che riesegui i test ottieni gli stessi numeri — e quindi gli stessi risultati da KMeans durante i test di split.
3. `mock_oracle_factory` è una factory: restituisce una funzione che costruisce un oracolo finto con risposte scriptate. Test diversi hanno bisogno di script diversi, quindi invece di una fixture fissa c'è una funzione che costruisce l'oracolo su misura.

---

**Otto file di test stub in `tests/phase2/`**

Un file per ogni modulo che dovrà esistere nella fase 2:
| File | Cosa testa |
|-----------|-----------|
| `test_feedback.py` | I 5 tipi di feedback e le costanti |
| `test_feedback_parser.py` | Il parser LLM del feedback |
| `test_uncertainty.py` | Il calcolo dell'entropia e dei candidati split/merge |
| `test_agent_functions.py` | Le funzioni core f_output, f_next_state, f_next_best_step |
| `test_hierarchy.py` | La genealogia dei cluster (split/merge history) |
| `test_oracle_protocol.py` | L'oracolo finto e il protocollo |
| `test_conversation_loop.py` | Il loop conversazionale da 30 turni |
| `test_app.py` | Le route Flask della web UI | 

Ogni test importa il codice dentro la funzione, non in cima al file:
```python
def test_split_feedback_construction():
    from src.feedback import SplitFeedback  # import qui dentro, non in cima
    fb = SplitFeedback(cluster_id=2, seed_item_ids=[10, 20])
    assert fb.cluster_id == 2
```

Se l'import fosse in cima al file, pytest crasherebbe già durante la raccolta dei test perché `src/feedback.py` non esiste ancora. Con l'import dentro la funzione invece, pytest raccoglie tutti i test regolarmente — li vede, li elenca — e solo quando li esegue ottiene `ImportError`. 
Questo è il RED state pulito cercato: i test esistono e falliscono con un errore preciso, non con un crash generico.

*Questa è stata una deviazione rispetto al piano originale, che prevedeva import in cima al file. GSD l'ha corretta durante l'esecuzione quando ha verificato che pytest crashava alla raccolta.* 

---

**Risultato della wave 0**

Alla fine della wave 0 il sistema è in questo stato:
- pytest trova 66 test ✓
- pytest li esegue: tutti rossi ✓

Questo è il punto di partenza. Ogni wave successiva prende un pezzo del sistema, lo costruisce, e trasforma alcuni di quei test rossi in verdi.

---

### Wave 1 — Feedback model

La wave 1 costruisce il linguaggio con cui l'oracle comunica al sistema cosa vuole cambiare. Prima di questa wave il sistema non aveva modo di capire "split cluster 0" o "sposta item 3 nel cluster 2" — erano solo parole. Dopo questa wave diventano oggetti Python precisi.

**Due file prodotti**

`src/feedback.py` — definisce 5 tipi di feedback come dataclass **frozen**:

| Dataclass | Campi | Quando si usa |
|-----------|-----------|-----------|
| `SplitFeedback` | `cluster_id`, `seed_item_ids` | "Dividi il cluster 0 in due" |
| `MergeFeedback` | `cluster_a_id`, `cluster_b_id` | "Unisci cluster 0 e 1" |
| `MoveItemFeedback` | `item_id`, `target_cluster_id` | "Sposta item 3 nel cluster 2" | 
| `GlobalFeedback` | `instruction_text` | "Troppi cluster in generale" |
| `InstructionalFeedback` | `instruction_text` | "Tratta X e Y come sinonimi" |

**Frozen** significa immutabili — una volta creati non si possono modificare. Se provi a scrivere `fb.cluster_id = 99` il programma crasha subito. Questo garantisce che il feedback non venga alterato accidentalmente mentre viene processato.

**Due costanti:**
- `ORACLE_MOVE_CONFIDENCE = 0.95` (quando sposti un item, il sistema è 95% sicuro della nuova assegnazione)
- `UNIFORM_FALLBACK_THRESHOLD = 1e-9` (guardia per un caso limite matematico nella redistribuzione delle probabilità).

`FeedbackDelta` è solo un nome comodo per dire "un feedback può essere uno di questi 5 tipi". Invece di scrivere ogni volta `SplitFeedback or MergeFeedback or MoveItemFeedback or...`, scrivi solo `FeedbackDelta`.<br>
L'ordine dei 5 tipi nell'unione non è casuale — GlobalFeedback viene messo primo perché quando il sistema riceve più feedback insieme, li processa in quest'ordine: prima i globali, poi split/merge, poi spostamenti singoli. L'ordine nell'alias riflette quest'ordine di priorità.

`src/feedback_parser.py` risolve un problema pratico: l'oracle scrive testo libero in italiano o inglese, ma il sistema ha bisogno di oggetti Python precisi. Il parser fa da traduttore.<br>
Funziona così: arriva il testo dell'oracle $\rightarrow$ viene mandato a Claude con un prompt che dice "trova i feedback in questo testo e restituiscili come JSON" $\rightarrow$ Claude risponde con un array JSON $\rightarrow$ il parser costruisce i dataclass corrispondenti.

Ad esempio se l'oracle scrive *"dividi il cluster 0 in due gruppi"*, Claude risponde:
```python 
[{"type": "split", "cluster_id": 0, "seed_item_ids": []}]
```

E il parser costruisce `SplitFeedback(cluster_id=0, seed_item_ids=[])`.

**Due guardie di sicurezza:**<br>
Se l'LLM restituisce un `cluster_id` che non esiste nello stato corrente, il programma crasha immediatamente. Se restituisce un tipo sconosciuto, stessa cosa. Questo impedisce che un'allucinazione dell'LLM causi danni silenziosi.

---

### Wave 2 — Moduli di supporto
La wave 2 costruisce quattro moduli che servono come fondamenta per il cuore del sistema. Nessuno di questi fa ancora girare la conversazione — preparano gli strumenti che la wave 3 userà.

**Quattro file prodotti**

1. **`src/hierarchy.py` — tiene traccia della storia dei cluster nel tempo.** <br><br>
Ogni volta che un cluster viene splittato o mergato, la sua storia va registrata. HierarchyStore è un dizionario che cresce nel tempo: ogni cluster ha un nodo con il suo ID, chi era il suo genitore, e se è ancora attivo o è stato sostituito. <br><br>
Le tre operazioni sono:
    - `register(cluster_id)` — aggiunge un nuovo cluster. Se lo aggiungi due volte crasha subito.
    - `record_split(parent_id, child_a_id, child_b_id)` — marca il cluster genitore come inattivo e registra i due figli.
    - `record_merge(parent_a_id, parent_b_id, merged_id)` — marca entrambi i genitori come inattivi e registra il cluster risultante. <br>

    Il punto chiave è HIER-02: la gerarchia parte vuota e cresce solo quando l'oracle fa un'azione. Non viene pre-calcolata all'inizio — questo è importante perché la struttura dei cluster cambia durante la conversazione.

---

2. **`src/uncertainty.py` — calcola quanto il sistema è incerto su ogni recensione e su ogni cluster.** <br><br>
`f_uncertainty` prende lo stato attuale e produce un UncertaintyReport con tre liste:
    - `boundary_items` — tutte le recensioni ordinate dalla più ambigua alla meno ambigua. Una recensione con `soft_probs = [0.5, 0.5]` è molto ambigua (non si sa dove metterla), una con `[0.95, 0.05]` è quasi certamente nel cluster 0.
    - `split_candidates` — i cluster ordinati da quello più "confuso" internamente (le sue recensioni hanno probabilità simili tra loro) a quello più compatto.
    - `merge_candidates` — tutte le coppie di cluster ordinate da quella più simile a quella più diversa.

    L'entropia usata è quella di Shannon normalizzata — un numero tra 0 e 1 dove 0 significa "certezza assoluta" e 1 significa "massima confusione". Dividere per `log(K)` serve a normalizzare: con 3 cluster il massimo teorico di entropia è diverso da quello con 10 cluster, e questa divisione li rende comparabili.

---

3. **`src/oracle_protocol.py` — definisce il contratto tra il sistema e l'oracle.**<br><br>
`OracleProtocol` è un'interfaccia che dice: qualsiasi oggetto che ha un metodo `reply(state, message) -> OracleReply` può fare da oracle. Nella fase 2 l'unico oracle è `MockOracle` — un oracle finto che segue uno script predefinito. Nella fase 3 verrà sostituito dall'oracle reale senza cambiare nulla nel resto del codice. <br><br>
`MockOracle` funziona come un attore con un copione: gli dai una lista di risposte in anticipo, e le restituisce una per turno. Quando finisce lo script restituisce una risposta neutra vuota. Se gli passi uno script vuoto crasha subito — non ha senso un oracle senza risposte.

---

4. **`src/strategy.py` — decide cosa mostrare o chiedere all'oracle ad ogni turno.**<br><br>
`RandomStrategy` è la strategia più semplice possibile: sceglie a caso tra le azioni disponibili. Le azioni possibili sono quattro: mostrare tutti i cluster, mostrarne un sottoinsieme, fare una domanda, o fermarsi.<br><br>
Il dettaglio importante è che usa `random.Random(seed)` — un'istanza privata del generatore casuale con seed fisso, non il generatore globale di Python. Questo garantisce che con lo stesso seed la strategia faccia sempre le stesse scelte — fondamentale per i test deterministici.

---

### Wave 3 – Agent core
Costruisce il cuore del sistema: le funzioni che applicano il feedback dell'oracle allo stato dei cluster, e il loop che fa girare la conversazione.

**`src/agent_functions.py` — tre funzioni pure che non fanno I/O, non scrivono file, non chiamano API. Ricevono uno stato, lo trasformano, restituiscono il nuovo stato.**
- `f_output(state)` — la funzione più semplice. Riceve lo stato attuale e lo restituisce invariato, dopo aver verificato che sia completo: tutte le recensioni hanno un'assegnazione, tutte hanno le probabilità, esiste almeno un cluster. È la garanzia che il sistema non restituisca mai uno stato parziale o rotto.

- `f_next_best_step(state, strategy, uncertainty_report)` — delega alla strategia la scelta dell'azione successiva. Non fa niente da sola — chiama `strategy.select()` e restituisce il risultato.

- `f_next_state(state, deltas, ...)` — la funzione più complessa. Prende lo stato attuale e una lista di feedback dell'oracle, e produce il nuovo stato applicandoli tutti. I feedback vengono processati in ordine di priorità fisso, indipendentemente dall'ordine in cui arrivano:
    1. `GlobalFeedback` per primo — istruzioni generali come "troppi cluster"
    
    2. `SplitFeedback` e `MergeFeedback` — operazioni strutturali sui cluster
    
    3. `MoveItemFeedback` — spostamenti di singole recensioni
    
    4. `InstructionalFeedback` per ultimo — istruzioni lessicali come "tratta X e Y come sinonimi"

    Questo ordine è fisso perché ha senso logico: prima si applicano le istruzioni globali, poi si riorganizza la struttura, poi si spostano i singoli item.

**Tre meccanismi interni:**

*Split* — quando l'oracle chiede di dividere un cluster, il sistema usa KMeans con K=2 sul sottoinsieme di embeddings di quel cluster. Se l'oracle ha indicato delle recensioni rappresentative dei due sottogruppi, vengono usate come centroidi iniziali. Altrimenti KMeans sceglie da solo. I due nuovi cluster ricevono ID monotonicamente crescenti — un ID ritirato non viene mai riusato.

*Merge* — quando due cluster vengono uniti, le loro probabilità vengono sommate colonna per colonna e poi rinormalizzate. Se un item aveva `[0.6, 0.3, 0.1]` e i cluster 0 e 1 vengono uniti, la nuova probabilità per il cluster risultante è `0.6 + 0.3 = 0.9`, poi tutto viene rinormalizzato a sommare 1.

*Move item* — quando una recensione viene spostata in un cluster, la sua probabilità per quel cluster viene fissata a `0.95` (la costante `ORACLE_MOVE_CONFIDENCE`). Il restante `0.05` viene distribuito proporzionalmente tra gli altri cluster. Se per qualche caso limite tutti gli altri cluster avevano probabilità 0, la distribuzione diventa uniforme grazie a `UNIFORM_FALLBACK_THRESHOLD`.

Il `global_instructions` è una lista che vive fuori dallo stato — lo schema di `ClusteringState` è congelato e non si può aggiungere un campo. Ogni `GlobalFeedback` appende il suo testo a questa lista, che viene passata a `f_next_state` ad ogni turno. Nella fase 3 questa lista verrà usata per arricchire i prompt di naming.

---

**`src/conversation_loop.py` — il loop che orchestra tutto.**

`run_conversation` è un semplice while True che ad ogni turno esegue questi passi in ordine:

1. Calcola l'incertezza con `f_uncertainty`

2. Sceglie l'azione con `f_next_best_step`

3. Formatta un messaggio e lo manda all'oracle

4. Riceve la risposta dell'oracle

5. Chiama `parse_feedback` per trasformarla in delta strutturati

6. Applica i delta con `f_next_state`

7. Scrive il nuovo stato su `audit_log.jsonl`

8. Emette l'aggiornamento via SocketIO al browser

9. Controlla le condizioni di stop

10. Avanza o esce dal loop

Due modalità di funzionamento: se `socketio=None` salta gli emit — utile nei test dove non c'è un server Flask in ascolto. Se `llm_client=None` salta il parsing — i delta sono sempre una lista vuota e lo stato non cambia mai.

---

*Il piano prevedeva di implementare `_apply_split` e `_apply_merge` come stub nella task 1 e completarli nella task 2. GSD ha invece implementato tutto subito nella task 1 perché i test di split e merge erano nello stesso file degli altri test — lasciarli come stub avrebbe fatto fallire i test che la task 1 doveva rendere verdi. Il risultato finale è identico, solo la sequenza interna è cambiata.*

*C'era anche un test con raise `NotImplementedError` nel corpo — `test_global_feedback_accumulates` — rimasto dalla wave 0 come stub intenzionale. GSD lo ha completato durante questa wave.*

**Risultato della wave 3**

18 test verdi tra `test_agent_functions.py` e `test_conversation_loop.py`. Il loop da 30 turni con `MockOracle` gira senza errori e l'audit log viene scritto correttamente ad ogni turno.

---

### Wave 4 — Web UI

La wave 4 costruisce l'interfaccia web che permette di osservare la conversazione in tempo reale dal browser mentre il loop gira in background.

**Il server (`web/app.py`)** gestisce tre cose:

- `GET /` — serve la pagina HTML

- `GET /status` — restituisce un JSON con lo stato della sessione: idle o in esecuzione, turno corrente, numero di cluster

- `POST /upload` — riceve il dataset, avvia la conversazione

Il punto più importante di `app.py `è come funziona l'upload. Quando carichi un file, il server risponde subito con 200 — non aspetta niente. Tutto il lavoro pesante (calcolo embeddings, clustering, loop conversazionale) parte in un thread separato in background. Questo è corretto perché il clustering su 12.000 recensioni richiede minuti — non puoi tenere il browser in attesa.

Due regole fisse nel codice che non si possono violare:

1. `debug=False` sempre — Flask in modalità debug espone un debugger accessibile dal browser, un rischio di sicurezza

2. Mai `from flask_socketio import emit` — questo emit è legato alla richiesta HTTP corrente e crasha se chiamato da un thread in background. Si usa sempre `socketio.emit()` come metodo dell'istanza

---

**La pagina** (`web/templates/index.html`) ha due colonne: a sinistra una griglia di card (una per cluster), a destra una sidebar con turno corrente, carico cognitivo e cronologia della conversazione.

---

**Il client WebSocket** (`web/static/main.js`) si connette al server e aggiorna la pagina automaticamente ogni volta che arriva un state_update — senza ricaricare la pagina. Per ogni cluster mostra le 5 recensioni con la confidenza più alta.

Dopo uno split o un merge, gli ID dei cluster diventano non consecutivi — per esempio 0, 2, 5. Se il codice leggesse le probabilità per posizione (prima colonna, seconda colonna...) darebbe valori sbagliati in silenzio. Invece usa sempre l'ID del cluster come chiave: `softProbs[itemId][cluster.id]`. Così è sempre corretto indipendentemente da quanti split o merge sono avvenuti.

---

**Il CSS** (`web/static/style.css`) implementa il layout a due colonne, le card dei cluster e una piccola barra colorata che mostra visivamente la confidenza di ogni recensione.

---
*Il piano prevedeva di creare prima `app.py` e poi i file statici in un secondo momento. Ma il test `test_index_route_returns_200` richiedeva già `index.html` durante la prima task. GSD ha creato tutti i file insieme.*

*Il piano metteva il clustering direttamente nel route handler dell'upload. GSD lo ha spostato nel thread in background perché altrimenti i test fallivano — il handler chiamava `assert api_key` e crashava con 500 invece di rispondere 200.*

*Alcuni commenti nel codice contenevano le stringhe `debug=True` e from `flask_socketio import emit` come esempi di cosa non fare. I grep dei criteri di accettazione le trovavano comunque e segnalavano errore. GSD ha riscritto i commenti.*

**Risultato della wave 4**

5 test verdi in `test_app.py`. Un warning atteso nei test: il thread in background crasha perché il pacchetto `anthropic` non è installato nell'ambiente di test — ma il crash avviene nel thread, non nel test, quindi i test passano lo stesso.

---

### Wave 5 — Backend multipli

La wave 5 aggiunge la possibilità di scegliere quale algoritmo di clustering usare, e lo rende selezionabile da riga di comando quando si avvia il server.

Un protocollo comune (`ClusteringBackend`) — prima di questa wave il sistema usava solo HDBSCAN, hardcoded. Ora esiste un'interfaccia comune che qualsiasi backend deve rispettare: un metodo `fit(embeddings)` che riceve gli embeddings e restituisce etichette e probabilità. Questo significa che aggiungere un terzo backend in futuro richiede solo implementare quel metodo — nient'altro nel sistema cambia.

**`HDBSCANBackend`** — avvolge il codice HDBSCAN già esistente dalla fase 1 dentro la nuova interfaccia. L'unica cosa che aggiunge è la garanzia che non escano mai etichette -1 (le recensioni "rumore" che HDBSCAN non riesce ad assegnare) — vengono sempre forzate nel cluster più vicino prima di restituire il risultato.

**`KMeansBackend`** — il backend nuovo. Ha due fasi:

1. **Selezione automatica di K** — KMeans richiede di sapere in anticipo quanti cluster creare, ma noi non lo sappiamo. Il sistema lo trova automaticamente usando il BIC (Bayesian Information Criterion): prova K=2, K=3, K=4... fino a K=√N, e sceglie il K che minimizza il BIC. In pratica sceglie il numero di cluster che spiega meglio i dati senza essere eccessivamente complesso. Per velocità usa la covarianza diagonale e massimo 50 iterazioni per ogni prova.

2. **Probabilità morbide** — KMeans di per sé assegna ogni recensione a un solo cluster senza incertezza. Per avere probabilità morbide come HDBSCAN, il sistema calcola la distanza di ogni recensione da ogni centroide e applica una softmax sulle distanze negative. Più una recensione è vicina a un centroide, più alta è la sua probabilità per quel cluster. La costante `KMEANS_SOFTMAX_TEMP = 1.0` controlla quanto sono "morbide" queste probabilità — con 1.0 le distanze vengono usate direttamente senza scalatura.

---

Flag `--backend` in `web/app.py` — quando si avvia il server si può scegliere:
```python
python web/app.py --backend kmeans
python web/app.py --backend hdbscan   # default
```

Il valore viene letto una volta sola all'avvio con `argparse`. Se si passa un valore sconosciuto il programma crasha immediatamente con un messaggio chiaro. Il backend scelto e il K utilizzato vengono scritti nell'`audit_log.jsonl` come evento `backend_init` — così ogni run è riproducibile sapendo esattamente con quale configurazione è stato eseguito.

---

**Risultato wave 5**

`src/clustering.py` ora esporta `ClusteringBackend`, `HDBSCANBackend`, `KMeansBackend` e `KMEANS_SOFTMAX_TEMP`. 

`build_initial_clustering_state` accetta un parametro opzionale backend — se non viene passato usa HDBSCAN come prima, garantendo compatibilità con tutto il codice esistente.

---

### Wave 6 — Proiezione UMAP
La wave 6 aggiunge una visualizzazione 2D dei cluster nel browser — un grafico a dispersione che mostra dove si trovano le recensioni nello spazio degli embeddings e come sono raggruppate.

Gli embeddings sono vettori da 768 numeri — impossibili da visualizzare direttamente. UMAP è un algoritmo che comprime quei 768 numeri in 2, cercando di mantenere le relazioni di vicinanza: recensioni simili restano vicine, recensioni diverse restano lontane. Il risultato è una mappa 2D dei dati che si può disegnare su un canvas.

**Tre funzioni in `web/app.py` :**

`_compute_projection(embeddings)` — esegue UMAP e restituisce le coordinate 2D. Usa sempre `random_state=42` per garantire che ogni volta che esegui il programma sullo stesso dataset ottenga esattamente le stesse coordinate — fondamentale per la riproducibilità degli esperimenti.

`_build_projection_payload(coords, state)` — costruisce il pacchetto dati da mandare al browser. Contiene le coordinate di ogni recensione, il suo cluster di appartenenza, la sua probabilità massima (usata per l'opacità del punto), e un colore per ogni cluster scelto da una palette di 20 colori distinti.

`_should_recompute_projection(deltas)` — decide se ricalcolare la proiezione dopo un turno. La risposta è sì solo se è avvenuto uno split o un merge — operazioni che cambiano il numero di cluster. Per uno spostamento singolo di una recensione non vale la pena ricalcolare — i punti non si sono mossi, è cambiata solo l'etichetta di colore.

---

`compute_and_emit_projection(store, state, socketio)` — orchestra le tre funzioni sopra e manda il risultato al browser via SocketIO come evento `projection_update`.

`post_turn_callback` in `src/conversation_loop.py` — per far ricalcolare la proiezione dopo ogni split o merge senza accoppiare il loop alla logica di visualizzazione, è stato aggiunto un parametro opzionale `post_turn_callback` a `run_conversation`. Dopo ogni turno, se questo parametro è definito, il loop lo chiama passando il nuovo stato e i delta. In `web/app.py` questo callback controlla se c'è stato uno split o merge e in quel caso ricalcola la proiezione. È backward-compatible — chi chiama `run_conversation` senza questo parametro non nota nessuna differenza.

---

**Nel browser** — `index.html` riceve un `<canvas>` a tutta larghezza sopra le due colonne. `main.js` gestisce l'evento `projection_update` e disegna i punti sul canvas: ogni punto ha il colore del suo cluster e un'opacità proporzionale alla sua probabilità massima — i punti più certi sono più opachi, quelli ambigui più trasparenti.

---

Nella fase 2 il `MockOracle` non produce mai `SplitFeedback` o `MergeFeedback` — le sue risposte sono sempre vuote. Quindi `_should_recompute_projection` restituisce sempre `False` e la proiezione viene calcolata solo una volta all'avvio, mai durante la conversazione. Il cablaggio è corretto e si attiverà automaticamente quando nella fase 3 arriverà l'oracle reale.

*Il summary di questa wave documenta un dettaglio: dopo aver confermato che i test erano rossi (RED state), GSD ha fatto un `git stash` per verificare dei fallimenti preesistenti. Lo stash ha rimosso le modifiche non ancora committate. Tutto è stato riapplicato e committato correttamente — nessun lavoro perso.*

**Risultato wave 6**

10 nuovi test verdi in `test_umap_projection.py`. I 5 test di `test_app.py` restano verdi.

---

### Wave 7 — Sessioni persistenti

La wave 7 aggiunge la persistenza delle sessioni: ogni conversazione viene salvata su disco e può essere ripresa in seguito, anche dopo aver riavviato il server.

Prima di questa wave, riavviare il server significava perdere tutto — stato dei cluster, cronologia dei turni, embeddings. Ogni volta bisognava ricaricare il dataset e ricominciare da zero. Con le sessioni persistenti ogni conversazione viene salvata in una cartella dedicata e può essere ripresa con un click.

---

**Una cartella per ogni sessione** — quando parte una nuova conversazione, il sistema crea una cartella con un timestamp come nome, tipo `sessions/2026-05-08T14-32-00/`. I due punti vengono sostituiti con trattini perché Windows non li ammette nei nomi di file. Dentro la cartella vengono salvati tre file: `state.json` con lo stato più recente, `audit_log.jsonl` con la cronologia completa dei turni, e `embeddings.npy` con gli embeddings del dataset caricato.

**Scrittura ad ogni turno** — `state.json` non viene scritto solo alla fine della conversazione, ma dopo ogni turno. Questo avviene tramite il `post_turn_callback` introdotto nella wave 6 — nella wave 7 questo callback fa due cose insieme: ricalcola la proiezione UMAP se necessario (wave 6) e scrive `state.json` (wave 7). I due compiti sono combinati in un unico callable `_per_turn_callback` passato a `run_conversation`.

---

**Due nuove route in `web/app.py`** :

`GET /sessions` — scansiona la cartella `sessions/`, legge il `state.json` di ogni sottocartella, e restituisce una lista ordinata dalla più recente con queste informazioni per ognuna: ID sessione, timestamp, numero di cluster, numero di turni. Se la cartella non esiste restituisce una lista vuota.

`POST /resume/<session_id>` — carica il `state.json` della sessione richiesta, ricostruisce il `ClusteringState`, lo imposta come stato corrente, e manda immediatamente un `state_update` via SocketIO così il browser si aggiorna. Se la cartella o il file non esistono, crasha subito con un messaggio chiaro — nessun fallback silenzioso.

---

**Nel browser** — la sidebar ora ha una sezione Sessions sotto la cronologia conversazione. Quando il browser si connette al server, carica automaticamente la lista delle sessioni passate. Cliccando su una sessione viene chiamato `POST /resume` e lo stato viene ripristinato nel browser via SocketIO.

---

*Flask in modalità test imposta `PROPAGATE_EXCEPTIONS=True` di default, che fa propagare le eccezioni fuori dal test client invece di restituire un codice di errore HTTP. Il test `test_resume_endpoint_exists` si aspettava un 500 quando si tenta di riprendere una sessione inesistente, ma invece riceveva l'eccezione direttamente. GSD ha aggiunto `app.config["PROPAGATE_EXCEPTIONS"] = False` nella fixture del test client, con un ripristino automatico al termine del test per non influenzare gli altri test.*

*I criteri di accettazione richiedevano che `loadSessionsList` apparisse almeno 3 volte nel codice JavaScript. Con la definizione della funzione e la chiamata al connect erano solo 2. GSD ha aggiunto una terza chiamata dentro `resumeSession` — dopo che una sessione viene ripresa con successo, la lista delle sessioni viene ricaricata per aggiornare il contatore dei turni mostrato in sidebar. Comportamento corretto oltre che necessario per superare il criterio.*

**Risultato wave 7**

10 nuovi test verdi in `test_sessions.py`. I test preesistenti di `test_app.py` e `test_umap_projection.py` restano verdi. Due fallimenti preesistenti documentati nel summary: `hdbscan` e `umap-learn` non installati nell'ambiente — risolti dalla wave successiva.

---

### Wave 8 — Fix lazy import e standardizzazione tipi

La wave 8 non aggiunge funzionalità nuove. Risolve due bug che bloccavano la suite di test.

**Bug 1 — import hdbscan a livello di modulo**

`src/clustering.py` aveva `import hdbscan` in cima al file. Questo significa che ogni volta che qualsiasi parte del codice importava qualcosa da `src/clustering.py` — anche solo `KMeansBackend` che non usa hdbscan — Python cercava il pacchetto `hdbscan` e crashava se non era installato.

La soluzione è spostare l'import dentro la funzione `run_hdbscan()`, che è l'unica che lo usa davvero:

```python
def run_hdbscan(embeddings, ...):
    import hdbscan  # ← caricato solo quando serve
    ...
```

Così from `src.clustering import KMeansBackend` funziona sempre. Solo chiamare `run_hdbscan()` o `HDBSCANBackend.fit()` richiede che hdbscan sia installato — e se non lo è, il programma crasha lì con un errore chiaro invece di crashare silenziosamente all'import.

---

**Bug 2 — tipo sbagliato su `SplitFeedback.seed_item_ids`**

`src/feedback.py` dichiarava `seed_item_ids: tuple[int, ...]` — una tupla. Ma `src/feedback_parser.py` costruiva l'oggetto con `tuple(item["seed_item_ids"])`, e i test usavano liste `[10, 20]` e confrontavano con liste. Tre posti diversi, tre contratti diversi.

La correzione ha standardizzato tutto su `list[int]`:

- `src/feedback.py` — annotazione cambiata da `tuple[int, ...]` a `list[int]`

- `src/feedback_parser.py` — `tuple(...)` cambiato in `list(...)`

- `tests/phase2/test_umap_projection.py` — `seed_item_ids=()` cambiato in `seed_item_ids=[]`

---

### Wave 9 — Dipendenze ambiente

La wave 9 risolve il problema delle dipendenze mancanti nell'ambiente Python.

**Installare `umap-learn`** — riuscito. Installato con le sue dipendenze (`numba`, `llvmlite`, `pynndescent`). I 10 test di `test_umap_projection.py` passano tutti verdi.

**Installare `hdbscan`** — fallito. Il pacchetto non ha wheel precompilati per Python 3.14 su Windows. La compilazione da sorgente richiede Microsoft C++ Build Tools che non era installato sulla macchina. Tutti i tentativi (`--only-binary`, `--no-build-isolation`, installazione da GitHub) hanno fallito con lo stesso errore.

----

**La soluzione adottata**

Invece di bloccare tutto su hdbscan, GSD ha aggiunto guardie `pytest.importorskip` in cima ai due file di test che dipendono da pacchetti opzionali:

```python
# in test_clustering_backends.py
pytest.importorskip("hdbscan", reason="pip install hdbscan per eseguire questi test")

# in test_umap_projection.py  
pytest.importorskip("umap", reason="pip install umap-learn per eseguire questi test")
```

Quando il pacchetto manca, pytest salta l'intero file con un messaggio chiaro invece di farlo crashare con `ModuleNotFoundError`. Questo è il pattern standard pytest per dipendenze opzionali — verde o skippato, mai rosso per colpa dell'ambiente.

---

**Risultato wave 9**
```
72 passed
13 failed  ← preesistenti, sentence_transformers non installato
1 skipped  ← test_clustering_backends.py (hdbscan mancante)
```

I 13 fallimenti preesistenti in `test_agent_functions.py` e `test_conversation_loop.py` sono causati da `sentence_transformers` non installato — fuori scope per questa wave.

## src

### `feedback.py`
Definisce i 5 tipi di feedback come dataclass frozen e due costanti.

**Frozen significa immutabili** — una volta creato un oggetto non si può modificare. Se provi a scrivere `fb.cluster_id = 99` il programma crasha subito. Questo garantisce che il feedback non venga alterato accidentalmente mentre viene processato dal sistema.

I 5 tipi:
| Dataclass | Campi | Quando si usa |
|-----------|-----------|-----------|
| `GlobalFeedback` | `instruction_text` | "Troppi cluster in generale" |
| `SplitFeedback` | `cluster_id`, `seed_item_ids` | "Dividi il cluster 0 in due" |
| `MergeFeedback` | `cluster_a_id`, `cluster_b_id` | "Unisci cluster 0 e 1" |
| `MoveItemFeedback` | `item_id`, `target_cluster_id` | "Sposta item 3 nel cluster 2" | 
| `InstructionalFeedback` | `instruction_text` | "Tratta X e Y come sinonimi" |

`FeedbackDelta` è un alias che significa "un feedback può essere uno qualsiasi di questi cinque tipi". L'ordine nell'alias non è casuale — riflette l'ordine in cui vengono processati: prima i globali, poi split/merge, poi spostamenti singoli, poi istruzioni lessicali.

Due costanti:
- `ORACLE_MOVE_CONFIDENCE = 0.95` — quando l'oracle sposta una recensione in un cluster, il sistema imposta la sua probabilità per quel cluster a 0.95

- `UNIFORM_FALLBACK_THRESHOLD = 1e-9` — guardia per un caso limite: se tutte le probabilità rimanenti sono zero dopo uno spostamento, la distribuzione diventa uniforme invece di crashare

---
### `feedback_parser.py`

Prende il testo grezzo dell'oracle e lo trasforma nei dataclass di `feedback.py` chiamando l'LLM.

**Il flusso principale in `parse_feedback` :**

1. Se il testo è vuoto, restituisce subito `[ ]` senza chiamare l'LLM

2. Costruisce un riassunto dei cluster correnti tipo `"0: Alpha, 1: Beta, 2: Gamma"`

3. Manda il testo dell'oracle a Claude con un prompt che dice "restituisci un array JSON con i feedback che trovi"

4. Pulisce la risposta rimuovendo eventuali backtick markdown

5. Fa il parse del JSON

6. Costruisce i dataclass corrispondenti

`_build_delta` è la funzione che trasforma un singolo item JSON in un dataclass. Contiene tre guardie di sicurezza:

- Se il tipo non è uno dei 5 validi, crasha subito

- Se un `cluster_id` non esiste nello stato corrente, crasha subito

- Se si tenta di mergiare un cluster con se stesso, crasha subito

Queste guardie impediscono che un'allucinazione dell'LLM causi danni silenziosi al sistema.

C'è esattamente un `try/except` in tutto il file — solo attorno a `json.loads`. È l'unico errore legittimo da gestire: se l'LLM restituisce JSON malformato il programma crasha con un errore chiaro. Tutto il resto fallisce ad alta voce senza essere intercettato.

---

### `hierarchy.py`

Tiene traccia della storia dei cluster nel tempo — ogni split e ogni merge viene registrato qui.

La struttura è semplice: `HierarchyStore` è un dizionario dove ogni cluster ha un nodo (`ClusterNode`) con quattro informazioni: il suo ID, chi era il suo genitore, i suoi figli, e se è ancora attivo o è stato sostituito.

Le tre operazioni:

- `register(cluster_id)` — aggiunge un nuovo cluster al dizionario. Se provi ad aggiungere due volte lo stesso ID crasha subito — ogni ID è unico e non viene mai riusato.

- `record_split(parent_id, child_a_id, child_b_id)` — registra uno split. Marca il cluster genitore come inattivo (`is_active = False`), salva i due figli nella sua lista `children_ids`, e registra entrambi i figli come nuovi nodi con `parent_id` che punta al genitore.

- `record_merge(parent_a_id, parent_b_id, merged_id)` — registra un merge. Marca entrambi i genitori come inattivi e registra il cluster risultante come figlio del primo genitore, per convenzione. Il secondo genitore non viene perso — è ancora nel dizionario, semplicemente inattivo.

Il punto chiave: `HierarchyStore` parte sempre vuoto e cresce solo quando l'oracle fa un'azione. Non viene pre-calcolato all'inizio della conversazione.

---
### `uncertainty.py`

Calcola quanto il sistema è incerto su ogni recensione e su ogni cluster, e restituisce tre liste ordinate.

`f_uncertainty` prende lo stato corrente e produce un `UncertaintyReport` con tre viste:

- `boundary_items` — tutte le recensioni ordinate dalla più ambigua alla meno ambigua. L'incertezza di ogni recensione viene calcolata con l'entropia di Shannon normalizzata. Una recensione con `soft_probs = [0.5, 0.5]` ha entropia massima (non si sa dove metterla), una con `[0.95, 0.05]` ha entropia quasi zero (quasi certamente nel cluster 0). Dividere per `log(K)` normalizza il risultato tra 0 e 1 indipendentemente da quanti cluster ci sono.

- `split_candidates` — i cluster ordinati dal più confuso al più compatto. Per ogni cluster viene calcolata la media dell'entropia delle sue recensioni. Un cluster con recensioni molto diverse tra loro avrà entropia media alta ed è un buon candidato per lo split.

- `merge_candidates` — tutte le coppie di cluster ordinate dalla più simile alla più diversa. La somiglianza viene calcolata come distanza euclidea tra i centroidi dei cluster nello spazio delle probabilità — non nello spazio degli embeddings. Due cluster con probabilità simili sono vicini e potrebbero essere uniti.

Una guardia importante: se un cluster non ha recensioni il programma crasha subito con un messaggio chiaro. Cluster vuoti non dovrebbero mai arrivare a `f_uncertainty`.

--- 

### `oracle_protocol.py`

Definisce il contratto tra il sistema e l'oracle, e fornisce un oracle finto per i test.

`OracleReply` è la struttura che l'oracle restituisce ad ogni turno. Ha tre campi: il testo della risposta, un booleano `satisfied` che dice se l'oracle è soddisfatto del clustering (condizione di stop primaria), e `turn_cognitive_loa`d che nella fase 2 è sempre 0.0 — la fase 3 lo calcolerà davvero. Non è frozen perché la fase 3 potrebbe aggiungere campi senza rompere il codice esistente.

`OracleProtocol` è un'interfaccia che dice: qualsiasi oggetto con un metodo `reply(state, message) -> OracleReply` può fare da oracle. Il decorator `@runtime_checkable` permette di verificare a runtime con `isinstance(obj, OracleProtocol)` se un oggetto rispetta l'interfaccia — senza che quell'oggetto debba ereditare da nulla.

`MockOracle` è l'oracle finto usato nella fase 2. Funziona come un attore con un copione: riceve una lista di risposte in anticipo e le restituisce una per turno nell'ordine dato. Quando lo script finisce restituisce una risposta neutra vuota invece di crashare — questo permette ai test del loop da 30 turni di girare anche con script più corti. Se gli passi uno script vuoto crasha subito — non ha senso un oracle senza nemmeno una risposta.

`MockOracle` non eredita da `OracleProtocol` — implementa semplicemente un metodo `reply` con la stessa firma. Grazie al `@runtime_checkable`, `isinstance(oracle, OracleProtocol)` restituisce `True` lo stesso.

---

### `strategy.py`

Decide quale azione compiere ad ogni turno della conversazione.

`Action` rappresenta una singola azione con due campi: il tipo (`show_full`, `show_subset`, `ask_question`, `stop`) e un payload opzionale. Nella fase 2 il payload è sempre un dizionario vuoto — la fase 5 lo arricchirà con informazioni specifiche come quale cluster mostrare o quale domanda fare.

`StrategyProtocol` è l'interfaccia per le strategie — qualsiasi oggetto con un metodo `select(state, uncertainty_report) -> Action` è una strategia valida. La fase 5 aggiungerà `UncertaintyDrivenStrategy` e `BoundaryDrivenStrategy` senza toccare questo file.

`_enumerate_valid_actions` costruisce la lista di azioni disponibili in base allo stato corrente. Le regole sono semplici: `show_full` e `stop` sono sempre disponibili, `ask_question` richiede almeno un cluster, `show_subset` richiede almeno due cluster. La lista non può mai essere vuota — se lo fosse il programma crasherebbe con un messaggio che dice esplicitamente "questo è un bug".

`RandomStrategy` è la strategia della fase 2: sceglie a caso tra le azioni disponibili. Usa `random.Random(seed)` — un'istanza privata del generatore casuale, non il generatore globale di Python. Questo è fondamentale per i test: con lo stesso seed la strategia fa sempre le stesse scelte, rendendo i test deterministici e riproducibili.

---

### `agent_functions.py`

Contiene le funzioni pure che trasformano lo stato dei cluster in risposta al feedback dell'oracle. Nessuna di queste funzioni fa I/O, scrive file o chiama API — ricevono uno stato, lo trasformano, restituiscono il nuovo stato.

`f_output(state)` — la più semplice. Riceve lo stato e lo restituisce invariato dopo tre controlli: che ci siano recensioni, che tutte abbiano le probabilità, che esista almeno un cluster. È la garanzia che il sistema non pubblichi mai uno stato incompleto.

`f_next_best_step(state, strategy, uncertainty_report)` — delega interamente alla strategia. Chiama `strategy.select()` e restituisce il risultato. Non fa nient'altro.

`f_next_state(state, deltas, ...)` — la più complessa. Riceve lo stato corrente e una lista di feedback, e produce il nuovo stato applicandoli tutti in ordine di priorità fisso: prima i `GlobalFeedback`, poi split e merge, poi spostamenti singoli, poi istruzioni lessicali. L'ordine è fisso indipendentemente da come arrivano i feedback.

**Tre meccanismi interni:**

*`_apply_split`* — divide un cluster in due usando KMeans con K=2 sul sottoinsieme di embeddings di quel cluster. Se l'oracle ha indicato recensioni rappresentative dei due sottogruppi, vengono usate come centroidi iniziali. Altrimenti KMeans sceglie da solo con k-means++. Dopo lo split le probabilità del cluster ritirato vengono distribuite integralmente al sottocluster corretto — se un item finisce nel sottocluster A, la sua probabilità precedente per il cluster padre va interamente ad A. Le probabilità degli item negli altri cluster vengono rinormalizzate. I due nuovi cluster ricevono ID monotonicamente crescenti — un ID ritirato non viene mai riusato. Alla fine viene verificato che ogni riga di probabilità sommi a 1.0.

*`_apply_merge`* — unisce due cluster in uno. Le probabilità vengono fatte con column pooling: per ogni recensione la nuova probabilità per il cluster risultante è la somma delle probabilità dei due cluster uniti. Tutto viene poi rinormalizzato. Il nuovo cluster riceve un ID nuovo, i due vecchi vengono rimossi.

*`_apply_move_item`* — sposta una recensione in un cluster. La sua probabilità per il cluster target viene fissata a 0.95. Il restante 0.05 viene distribuito proporzionalmente tra gli altri cluster. Se tutti gli altri cluster avevano probabilità zero, la distribuzione diventa uniforme — questo è il caso limite gestito da `UNIFORM_FALLBACK_THRESHOLD`. C'è anche un no-op guard: se l'LLM suggerisce di spostare una recensione nel cluster in cui è già, lo stato viene restituito invariato.

`global_instructions` — è una lista che vive fuori dallo stato. Lo schema di ClusteringState è congelato e non si può modificare. Ogni `GlobalFeedback` appende il suo testo a questa lista, che persiste per tutta la sessione. Nella fase 3 verrà usata per arricchire i prompt di naming con le preferenze accumulate dell'oracle.

Dopo ogni operazione `f_next_state` verifica tre invarianti: il numero di recensioni non è cambiato, tutte hanno le probabilità, esiste ancora almeno un cluster. Se uno di questi è violato il programma crasha immediatamente.

---

### `conversation_loop.py`

Orchestra tutto il sistema in un semplice `while True`. È l'unico posto che fa I/O — scrive il log, emette eventi SocketIO. Le funzioni pure non scrivono nulla.

`run_conversation` accetta tutti i componenti come parametri e li usa in sequenza ad ogni turno:

1. Calcola l'incertezza con `f_uncertainty`

2. Sceglie l'azione con `f_next_best_step`

3. Formatta un messaggio leggibile per l'oracle con `_format_message`

4. Chiede la risposta all'oracle

5. Trasforma la risposta in delta strutturati con `parse_feedback` — solo se c'è un `llm_client` e la risposta non è vuota

5. Applica i delta con `f_next_state`

6. Scrive il nuovo stato su audit_log.jsonl

7. Chiama il `post_turn_callback` se definito — usato per ricalcolare la proiezione UMAP e scrivere `state.json` nelle sessioni persistenti

8. Emette `state_update` via SocketIO al browser — saltato se `socketio=None`

9. Controlla le condizioni di stop

10. Avanza o esce

Tre default utili: se `strategy=None` usa `RandomStrategy(seed=0)`, se criteria=None usa il budget di 15 turni, se `id_to_text=None` usa un dizionario vuoto. Questo permette di chiamare `run_conversation` nei test con il minimo indispensabile senza dover preparare tutto.

`global_instructions` viene inizializzata come lista vuota all'inizio della sessione e passata a `f_next_state` ad ogni turno — persiste per tutta la conversazione accumulando le istruzioni globali dell'oracle.

## web/

### `web/app.py`
È il server centrale che coordina tutto. Si divide in quattro aree principali.

**Route HTTP**
`GET /` — serve la pagina HTML. `GET /status` — restituisce lo stato della sessione corrente in JSON. `POST /upload` — riceve il dataset, lo valida, e avvia il task in background. `GET /sessions` — scansiona la cartella `sessions/` e restituisce la lista delle sessioni passate. `POST /resume/<session_id>` — carica una sessione salvata, imposta lo stato corrente, e manda subito un `state_update` al browser via SocketIO.

---

**Task in background**

`_run_conversation_background` è il cuore del server. Viene avviato da `/upload` e gira in un thread separato. Fa tutto il lavoro pesante in sequenza: calcola gli embeddings, crea la cartella sessione, istanzia il backend scelto (`hdbscan` o `kmeans`), costruisce lo stato iniziale, logga il backend usato nell'`audit_log.jsonl`, emette la proiezione UMAP iniziale, e avvia `run_conversation`. Usa sempre `socketio.emit()` come metodo dell'istanza — mai il `emit` importato dal modulo, che crasha nei thread in background.

---

**Helpers UMAP**

`_compute_projection` — esegue UMAP con `random_state=42` per garantire coordinate sempre identiche sullo stesso dataset. `_build_projection_payload` — costruisce il pacchetto dati da mandare al browser con coordinate, cluster di appartenenza, probabilità massime e colori. `_should_recompute_projection` — restituisce True solo se tra i delta c'è uno split o un merge. `compute_and_emit_projection` — orchestra le tre funzioni e manda il risultato via SocketIO.

---

**Helpers sessioni**

`_make_session_timestamp` — genera un timestamp con trattini invece dei due punti (`2026-05-08T14-32-00`) perché Windows non ammette i due punti nei nomi di file. `_write_session_state` — scrive il `ClusteringState` corrente in `state.json` dentro la cartella sessione, sovrascrivendo ad ogni turno. Viene chiamata tramite `_per_turn_callback` dopo ogni turno — lo stesso timing della scrittura JSONL.

---

### `web/templates/index.html`

Struttura a tre sezioni verticali. In cima l'header con il banner di stato e il form di upload. Sotto, a tutta larghezza, il canvas UMAP. Infine il layout a due colonne: a sinistra la griglia delle card dei cluster, a destra la sidebar con metriche, cronologia conversazione e lista sessioni.

---

### `web/static/main.js`

Gestisce tutti gli eventi WebSocket e aggiorna la pagina senza ricaricarla.

**Sessioni** — `loadSessionsList` viene chiamata alla connessione e dopo ogni resume. `renderSessionsList` costruisce la lista cliccabile. `resumeSession` chiama `POST /resume` e aggiorna il banner di stato.

**Aggiornamento stato** — `socket.on('state_update')` aggiorna turno, carico cognitivo, griglia card e cronologia. `renderTopItems` mostra le 5 recensioni più confidenti per ogni cluster usando sempre `softProbs[String(itemId)][String(cluster.id)]` — con l'ID del cluster come chiave, non la posizione, che sarebbe sbagliata dopo split e merge.

**Proiezione UMAP** — `socket.on('projection_update')` chiama `drawProjection` che normalizza le coordinate nel bounding box del canvas e disegna ogni punto come un cerchio colorato. L'opacità va da 0.3 a 1.0 in base alla probabilità massima dell'item — i punti certi sono più opachi, quelli ambigui più trasparenti. `hexToRgba` converte il colore esadecimale in rgba per applicare l'opacità.

---

### `web/static/style.css`

Layout a due colonne con CSS grid (`1fr 300px`). Le card dei cluster usano `auto-fill` con larghezza minima 220px — si adattano automaticamente alla larghezza dello schermo. La sidebar è sticky — resta visibile anche scorrendo. Il canvas UMAP è a tutta larghezza sopra il layout. Le `session-item` hanno hover blu chiaro e mostrano timestamp in grassetto e metadati in grigio sotto.

## Modifiche ai file esistenti dalla fase 1

#### `src/clustering.py`

È il file più modificato. Nella fase 1 conteneva solo `run_hdbscan`, `assign_noise_to_nearest` e `build_initial_clustering_state`. Nella fase 2 sono stati aggiunti:

- `KMEANS_SOFTMAX_TEMP = 1.0` — costante per il calcolo delle probabilità morbide di KMeans

- `ClusteringBackend` — protocollo comune che qualsiasi backend deve rispettare: un metodo fit(embeddings) che restituisce etichette e probabilità

- `HDBSCANBackend` — avvolge `run_hdbscan` e `assign_noise_to_nearest` nell'interfaccia comune, garantendo che non escano mai etichette -1

- `KMeansBackend` — backend nuovo con selezione automatica di K via BIC e probabilità morbide via softmax

Due modifiche ai file esistenti:

`import hdbscan` è stato spostato dalla cima del file dentro il corpo di `run_hdbscan()`. Prima della modifica, qualsiasi import da `src/clustering.py` richiedeva hdbscan installato — anche importare solo `KMeansBackend` che non lo usa. Dopo la modifica il modulo importa sempre, e hdbscan viene cercato solo quando `run_hdbscan()` viene effettivamente chiamato.

`build_initial_clustering_state` ha ricevuto un parametro opzionale backend: `ClusteringBackend | None = None`. Quando non viene passato usa `HDBSCANBackend` come default — tutto il codice esistente continua a funzionare senza modifiche.

---

#### `src/conversation_loop.py`

Un solo parametro aggiunto a `run_conversation`:

```python
post_turn_callback: Optional[Callable] = None
````

Viene chiamato dopo ogni turno con `(new_state, deltas)`. Nella fase 2 viene usato per due cose combinate in un unico callable: scrivere `state.json` nella cartella sessione e ricalcolare la proiezione UMAP se c'è stato uno split o un merge. Il default è `None` — tutto il codice esistente che chiama `run_conversation` senza questo parametro non nota nessuna differenza.
# Designing Large Scale AIS – Phase 3

## Pacchetti

La fase 3 non aggiunge pacchetti nuovi. Usa solo la libreria standard Python e il client Anthropic già presente dalla fase 2.

### Libreria standard Python aggiunta
1. `collections.deque` → `src/oracle_agent.py` – finestra scorrevole degli ultimi 10 feedback per il rilevamento contraddizioni
2. `datetime` → `src/oracle_agent.py` – timestamp dell'evento `oracle_init`
3. `pathlib.Path` → `src/oracle_agent.py` – gestione del percorso del file `events.jsonl`

---

## Wave

### Wave 0 – Scaffold dei test

Lo scopo della wave 0 è lo stesso delle fasi precedenti: scrivere tutti i test prima che esista qualsiasi implementazione. TDD (Test-Driven Development) puro — prima si definisce cosa il sistema deve fare, poi lo si costruisce.

**Una nuova fixture in `tests/conftest.py`:**

`oracle_agent_factory` è una factory che costruisce un `OracleAgent` con un client LLM finto (MagicMock). Funziona come `mock_oracle_factory` nella fase 2 — invece di un oracle con risposte scriptate, qui si ha un oracle LLM con un client che risponde sempre con un testo fisso. Il parametro `reply_text` permette a test diversi di simulare risposte diverse. La factory accetta anche i parametri di rumore (`consistency_rate`, `drift_probability`, `sycophancy_resistance`) per testare configurazioni diverse.

---

**Tre nuovi file di test in `tests/phase3/`:**

| File | Cosa testa |
|-----------|-----------|
| `test_oracle_agent.py` | OracleSpec, NoiseParams, OracleAgent e il rilevamento contraddizioni (13 test) |
| `test_cognitive_load.py` | La funzione `f_cognitive_load` (5 test) |
| `test_oracle_loop_integration.py` | L'OracleAgent collegato al loop conversazionale (3 test) |

Anche qui gli import sono dentro le funzioni di test, non in cima al file — stesso principio della fase 2: pytest raccoglie i test senza crashare, e li esegue trovando `ImportError` finché i file `src/` non esistono.

---

**Risultato della wave 0**

Alla fine della wave 0:
- pytest trova 21 nuovi test
- Tutti rossi con `ImportError` — atteso

---

### Wave 1 – Plan 01: OracleAgent, OracleSpec, NoiseParams

La wave 1 costruisce il cuore della fase 3: l'oracle LLM vero. Nella fase 2 l'unico oracle era `MockOracle` — un attore con un copione fisso. Da questa wave in poi esiste un oracle reale che chiama Claude, con preferenze configurabili e comportamento variabile.

**Un file nuovo, una modifica:**

`src/oracle_agent.py` — il file principale della fase 3. Contiene tre strutture dati e una classe:

`OracleSpec` è la specifica delle preferenze dell'oracle. Ha tre campi: quanti cluster vuole (`preferred_k`), su quali dimensioni raggruppa i dati (`semantic_axes`, ad esempio `["topic", "sentiment"]`), e una descrizione della persona che simula (`persona_description`). Questi tre campi vengono iniettati nel system prompt dell'LLM ogni volta che l'oracle risponde. Una volta creato un `OracleAgent`, la sua `OracleSpec` non cambia mai — se vuoi testare un oracle con preferenze diverse, crei un altro `OracleAgent`.

`NoiseParams` controlla quanto l'oracle è "umano" — cioè imperfetto e variabile. Tre parametri, tutti tra 0.0 e 1.0:
- `consistency_rate` — quanto spesso l'oracle accetta il clustering proposto senza chiedere cambiamenti. Con 0.8 accetta l'80% delle volte.
- `drift_probability` — con che probabilità introduce una nuova preferenza ad ogni turno, anche se contraddice qualcosa detto prima.
- `sycophancy_resistance` — quanto l'oracle mantiene la sua posizione quando il sistema non è d'accordo. Con 0.9 cede solo il 10% delle volte.

Tutti e tre diventano istruzioni in linguaggio naturale nel system prompt. Non c'è nessuna manipolazione della temperatura o post-processing — Claude riceve solo testo e si comporta di conseguenza.

`OracleAgent` è la classe principale. Si costruisce con `OracleSpec`, `NoiseParams` e il client LLM. Ad ogni turno, `reply()` assembla un system prompt a cinque sezioni e chiama il client:

1. Persona e preferenze (da `OracleSpec`)
2. Regole comportamentali (da `NoiseParams`)
3. Istruzioni accumulate nei turni precedenti (se ci sono)
4. Riassunto dello stato corrente (quanti cluster, quante item ciascuno)
5. Istruzione di overload, se il carico cognitivo è troppo alto (aggiunta dall'ultima sezione)

L'oracle è soddisfatto quando la sua risposta contiene il token `[SATISFIED]`. Il loop conversazionale usa questo come segnale di stop.

`src/oracle_protocol.py` viene modificato: `OracleReply` riceve due nuovi campi opzionali: `contradiction_detected: bool = False` e `contradicted_turn: int | None = None`. Sono facoltativi con valori di default — il `MockOracle` della fase 2 non deve cambiare nulla.

---

### Wave 1 – Plan 02: f_cognitive_load

La wave 1 Plan 02 costruisce la funzione che misura quanto è "pesante" ogni turno per l'oracle — quante informazioni deve elaborare contemporaneamente.

**Un file nuovo:**

`src/cognitive_load.py` — una funzione pura, 58 righe. `f_cognitive_load` riceve lo stato corrente e il messaggio del turno, e restituisce un numero tra 0 e 1.

La formula combina tre fattori con peso uguale:
- **Numero di cluster** rispetto al massimo (`MAX_K = 20`) — più cluster ci sono, più è difficile tenerli tutti in testa
- **Items mostrati** rispetto al totale — `f_next_best_step` mostra le prime 5 recensioni per cluster; più cluster ci sono, più recensioni vengono mostrate
- **Lunghezza del messaggio** rispetto al massimo (`MAX_MSG_LEN = 500`) — un messaggio lungo è più difficile da leggere

Se il risultato supera `COG_LOAD_THRESHOLD = 0.7`, il loop aggiunge al system prompt la frase `"OVERLOAD: Focus on one thing only."` — l'oracle risponde con un feedback più semplice e focalizzato invece di fare più richieste contemporaneamente.

Due guardie: se lo stato non ha cluster o non ha recensioni, il programma crasha subito. Cluster o stati vuoti non dovrebbero mai arrivare qui.

---

### Wave 2 – Drift detection

La wave 2 aggiunge il rilevamento delle contraddizioni. Il problema da risolvere: se al turno 3 l'oracle dice "dividi il cluster 0" e al turno 7 dice "unisci cluster 0 e 1", il sistema deve accorgersene.

**Una modifica a `src/oracle_agent.py`:**

`_contradicts(new_delta, prior_delta)` — funzione privata che confronta due feedback strutturalmente. Tre regole:
- Un `MergeFeedback(A, B)` contraddice un precedente `SplitFeedback` sullo stesso cluster A o B
- Un `SplitFeedback(X)` contraddice un precedente `MergeFeedback` che aveva X come input
- Un `MoveItemFeedback(item, target=B)` contraddice un precedente `MoveItemFeedback` sullo stesso item ma con destinazione diversa

`GlobalFeedback` e `InstructionalFeedback` vengono ignorati — sono troppo semantici per essere confrontati strutturalmente.

`update_delta_window(deltas, turn_index)` — il metodo chiamato dal loop dopo ogni turno. Funziona in due fasi:

1. **Prima controlla**: confronta ogni nuovo feedback con i 10 feedback precedenti nella finestra scorrevole (`deque(maxlen=10)`)
2. **Poi aggiunge**: appende i nuovi feedback alla finestra

L'ordine è importante: controllare prima di aggiungere evita che due feedback dello stesso turno si contraddicano tra loro. Se viene trovata una contraddizione, vengono restituiti `(True, turno_precedente)` — il loop poi scrive un `drift_event` nel file `events.jsonl`.

---

### Wave 3 – Wiring nel conversation loop

La wave 3 collega tutto al loop conversazionale. I tre moduli costruiti nelle wave precedenti esistevano ma erano isolati — questa wave li integra in `run_conversation`.

**Modifiche a `src/conversation_loop.py`:**

`_write_event(record, events_path)` — nuova funzione helper che scrive un record JSON su `events.jsonl`. È separata da `audit_log.jsonl` perché quel file contiene solo stati del clustering, e `load_audit_log()` crasherebbe se trovasse un record di tipo diverso.

`events_path` — nuovo parametro opzionale di `run_conversation`. Se non viene passato, il file viene creato automaticamente nella stessa cartella del log.

**Cinque aggiunte al loop:**

1. **`oracle_init`** — all'avvio, se l'oracle è un `OracleAgent`, il loop scrive su `events.jsonl` un record con tutti i parametri: `preferred_k`, `semantic_axes`, `consistency_rate`, `drift_probability`, `sycophancy_resistance`. Questo rende ogni run riproducibile — si sa esattamente con quale configurazione è stato eseguito.

2. **`f_cognitive_load`** — prima di chiamare `oracle.reply()` ad ogni turno, il loop calcola il carico cognitivo. Il valore viene passato direttamente a `reply()` invece di essere ricalcolato internamente — questo era il bug trovato in verifica (vedi sotto).

3. **`global_instructions` all'oracle** — se l'oracle è un `OracleAgent`, il loop gli passa la lista di istruzioni accumulate dai `GlobalFeedback` precedenti. L'oracle le include nel suo system prompt alla sezione 3.

4. **`update_delta_window`** — dopo che `f_next_state` ha applicato i feedback, il loop chiama `oracle.update_delta_window(deltas, new_state.turn_index)` per aggiornare la finestra e rilevare eventuali contraddizioni. Usa `new_state.turn_index` (dopo l'aggiornamento), non `state.turn_index` — altrimenti il turno registrato nella finestra sarebbe sfasato di uno.

5. **`drift_event`** — se `update_delta_window` rileva una contraddizione, il loop scrive su `events.jsonl` il record con il turno corrente e il turno della contraddizione precedente.

**Modifica a `src/agent_functions.py`:**

`f_next_state` ora appende il testo degli `InstructionalFeedback` alla lista `global_instructions` in-place. Prima i feedback istruzionali venivano ignorati — ora vengono accumulati e passati all'oracle nel turno successivo.

---

**Fix post-verifica (ORC-03)**

Dopo la prima verifica, un problema: il valore di `cognitive_load` calcolato dal loop veniva scartato. `reply()` non accettava il valore come parametro e lo ricalcolava internamente da zero — il calcolo del loop era codice morto. Il fix: aggiunto `cognitive_load: float | None = None` come parametro a `reply()`. Se viene passato, lo usa direttamente; se non viene passato, lo calcola da solo come fallback. Il loop ora lo passa sempre esplicitamente.

---

**Risultato della fase 3**

```
106 test passati (phase 1 + 2 + 3)
21 test nuovi in tests/phase3/
0 regressioni
3 fallimenti preesistenti in phase1 (hdbscan non installato — fuori scope)
```

5/5 requirements verificati: ORC-01 ✓ ORC-02 ✓ ORC-03 ✓ ORC-04 ✓ FB-04 ✓

---

## src

### `oracle_agent.py`

Il file più importante della fase 3. Contiene tutto quello che serve per simulare un oracle umano con un LLM.

**`OracleSpec`** — dataclass con tre campi: quanti cluster preferisce l'oracle (`preferred_k`), su quali dimensioni ragruppa i dati (`semantic_axes`), e la descrizione della sua personalità (`persona_description`). Questi tre campi vengono iniettati nel system prompt dell'LLM. La spec è fissa per tutta la vita dell'agente — per un oracle con preferenze diverse serve una nuova istanza.

**`NoiseParams`** — dataclass con tre parametri di rumore, tutti tra 0.0 e 1.0. Vengono validati all'inizializzazione con `assert` — se passi 1.5 il programma crasha subito. Anche questi diventano istruzioni nel system prompt, senza nessuna manipolazione tecnica della temperatura o della risposta.

**`OracleAgent`** — la classe principale. Al momento della costruzione:
- Valida i tre parametri di rumore
- Crea la `deque(maxlen=10)` per la finestra scorrevole del drift detection
- Scrive l'evento `oracle_init` su `events.jsonl` se è stato passato un path (utile per i test unitari che non passano dal loop)

Il metodo `_build_system_prompt(state, cognitive_load, global_instructions)` assembla il prompt a cinque sezioni in ordine:
1. Persona + preferenze + istruzione sul token `[SATISFIED]`
2. Regole comportamentali da `NoiseParams`
3. Istruzioni accumulate (solo se la lista non è vuota)
4. Riassunto dello stato corrente
5. `"OVERLOAD: Focus on one thing only."` — aggiunta solo se `cognitive_load > 0.7`

Il metodo `reply(state, message, global_instructions, cognitive_load)` assembla il prompt, chiama il client LLM, e restituisce un `OracleReply`. C'è una gestione speciale per il client Anthropic: l'SDK Anthropic accetta il system prompt come kwarg separato (`system=`), mentre gli altri adapter lo vogliono concatenato al messaggio. Il codice rileva il tipo di client e adatta la chiamata. `try/except` solo attorno alle chiamate LLM — tutto il resto fallisce ad alta voce.

`_contradicts(new_delta, prior_delta)` — funzione a livello di modulo (non metodo) che confronta due feedback. Tre regole strutturali: merge contraddice split sugli stessi cluster, split contraddice merge sugli stessi input, move su stesso item con destinazione diversa.

`update_delta_window(deltas, turn_index)` — controlla prima, aggiunge dopo. Restituisce `(True, turno_precedente)` alla prima contraddizione trovata, oppure `(False, None)` se tutto è coerente. `GlobalFeedback` e `InstructionalFeedback` vengono saltati — non si confrontano strutturalmente.

---

### `cognitive_load.py`

Funzione pura da 58 righe. Nessun I/O, nessuno stato globale.

**Quattro costanti:**
| Costante | Valore | Significato |
|----------|--------|-------------|
| `MAX_K` | 20 | Numero massimo di cluster per la normalizzazione |
| `MAX_MSG_LEN` | 500 | Lunghezza massima del messaggio per la normalizzazione |
| `TOP_K_ITEMS_PER_CLUSTER` | 5 | Quante recensioni vengono mostrate per cluster (allineato con `_format_message`) |
| `COG_LOAD_THRESHOLD` | 0.7 | Sopra questa soglia viene iniettata l'istruzione OVERLOAD |

**`f_cognitive_load(state, message)`** — formula a tre termini con peso uguale (1/3 ciascuno):
- `min(n_cluster / MAX_K, 1.0) × 1/3`
- `min((n_cluster × 5) / n_items, 1.0) × 1/3`
- `min(len(message) / MAX_MSG_LEN, 1.0) × 1/3`

Il risultato è sempre in `[0.0, 1.0]`. I `min(... , 1.0)` evitano che un caso estremo mandi fuori scala il risultato.

Due guardie: se lo stato non ha cluster o non ha items → `AssertionError` immediato.

---

## Modifiche ai file esistenti dalla fase 2

### `src/oracle_protocol.py`

`OracleReply` ha due nuovi campi, entrambi con valori di default:
- `contradiction_detected: bool = False`
- `contradicted_turn: int | None = None`

Non è frozen, quindi si possono aggiungere campi. Il `MockOracle` della fase 2 non deve cambiare nulla — i nuovi campi prendono semplicemente il valore di default.

---

### `src/conversation_loop.py`

Un nuovo parametro a `run_conversation`:

```python
events_path: str | None = None
```

Se non viene passato, il file `events.jsonl` viene creato automaticamente nella stessa cartella dell'`audit_log.jsonl`.

Cinque aggiunte nel corpo della funzione:
1. Import di `f_cognitive_load` a livello di funzione, fuori dal while loop
2. Scrittura `oracle_init` all'avvio se l'oracle è un `OracleAgent`
3. Calcolo `cognitive_load = f_cognitive_load(state, message)` ad ogni turno, prima di `oracle.reply()`
4. Chiamata `oracle.reply(..., global_instructions=global_instructions, cognitive_load=cognitive_load)` — solo se l'oracle è un `OracleAgent`
5. Chiamata `oracle.update_delta_window(deltas, new_state.turn_index)` dopo `f_next_state`
6. Scrittura `drift_event` su `events.jsonl` se viene rilevata una contraddizione

---

### `src/agent_functions.py`

Una sola aggiunta in `f_next_state`: quando arriva un `InstructionalFeedback`, il suo testo viene appeso alla lista `global_instructions` in-place:

```python
# FB-04: accumulate instructional feedback
if isinstance(delta, InstructionalFeedback):
    global_instructions.append(delta.instruction_text)
```

Prima i feedback istruzionali ("tratta X e Y come sinonimi") venivano ignorati. Ora vengono accumulati nella lista che il loop passa all'oracle ad ogni turno successivo.
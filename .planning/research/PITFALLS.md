# Domain Pitfalls: Conversational HITL Clustering

**Domain:** Conversational human-in-the-loop clustering / interactive ML research system
**Researched:** 2026-04-29
**Confidence:** HIGH for pitfall identification; MEDIUM for specific mitigations (domain is partially novel)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalidate claims, or make the research indefensible.

---

### Pitfall 1: Evaluation Measures What Clustering Does by Default, Not What Dialogue Adds

**What goes wrong:**
The system is evaluated on final clustering quality (NMI, ARI, silhouette score) or oracle satisfaction without a no-dialogue baseline. Any improvement appears to come from the conversation loop, but the underlying embedding and clustering algorithm may already produce the same result without interaction. The dialogue becomes decorative.

**Why it happens:**
Researchers build the interaction loop first, then reach for standard clustering metrics at evaluation time. Because the loop eventually produces a "good" clustering, the metric looks positive — but the metric was never designed to isolate the dialogue's contribution.

**Consequences:**
- No defensible headline claim. Reviewers will ask "does the interaction actually help?" and there is no answer.
- The core research contribution (efficient preference extraction) is untestable.
- The system reduces to an expensive wrapper around k-means.

**Warning signs:**
- Evaluation plan lists only clustering-quality metrics (NMI, ARI, silhouette) with no baseline condition.
- No experiment where the oracle gives zero feedback (cold-start baseline).
- Metrics do not track turns, cognitive load, or convergence rate.

**Prevention:**
- Define at minimum three conditions: (A) no dialogue (one-shot clustering), (B) random interaction (uninformative oracle), (C) full dialogue system.
- Primary metrics must be turn-efficiency (turns to reach oracle satisfaction threshold), cognitive load per turn (tokens read / decisions asked), and satisfaction at fixed turn budgets — not just final-state clustering quality.
- Run ablation: remove `f_next_best_step` (replace with random next-action policy) and measure degradation. If metrics do not degrade, the policy is not contributing.
- The Judge Agent's `f_eval` must be defined before any interaction code is written.

**Phase mapping:** Addressed in the evaluation design phase, before implementing the full interaction loop. If deferred, it becomes unfixable without a complete experiment redesign.

---

### Pitfall 2: LLM-Simulated Oracle Is Too Consistent and Too Helpful

**What goes wrong:**
The Oracle Agent, prompted to simulate a human with preferences, produces feedback that is:
- Perfectly consistent across turns (no realistic ambiguity or preference evolution)
- Never contradictory (humans contradict themselves ~15-30% of the time in annotation tasks)
- Over-cooperative (immediately accepts reasonable proposals rather than exploring alternatives)
- Sycophantic (agrees with the Clustering Agent's framing rather than asserting independent preferences)

The system is then tuned and evaluated against this unrealistically helpful oracle. When real humans interact, the system fails.

**Why it happens:**
LLMs exhibit documented sycophancy rates of ~58% in simulated evaluation scenarios (SycEval, 2025). RLHF-aligned models are trained to be agreeable. When prompted to play a "satisfied user," they lean toward satisfaction. The oracle persona specification rarely includes explicit instruction to be difficult, ambiguous, or to drift preferences over time.

**Root cause from research:**
LLM agents representing groups "display far less variance" than real humans — a "flattening effect." RLHF training decreases linguistic diversity and suppresses natural conversational variance. A simulated oracle will converge faster and more smoothly than a real person because it lacks the fatigue, distraction, inconsistency, and genuinely subjective preference structures that humans exhibit.

**Consequences:**
- Turns-to-convergence metric is systematically optimistic.
- Contradiction-handling logic (`latest intent wins`) is never stress-tested because the oracle never contradicts itself.
- System cannot generalize to real humans; the small human study will reveal the gap at the worst possible time (late in the project).

**Warning signs:**
- Oracle Agent has no explicit "noise" or "ambiguity" parameter.
- Oracle accepts > 80% of proposals in simulation runs.
- Preference profile never evolves across a session.
- Oracle never asks a clarifying question unprompted.

**Prevention:**
- Build the Oracle Agent with explicit configurable parameters: `consistency_rate` (0.0–1.0), `preference_drift_probability` (per-turn probability of evolving a stated preference), `sycophancy_resistance` (probability of rejecting a reasonable proposal to explore alternatives), `fatigue_model` (increasing brevity / decreasing engagement after N turns).
- Include at least one "adversarial oracle" persona: contradictory preferences, high drift, low cooperation.
- Run the human validation study (N = 5-10) before finalizing any quantitative claims, not after. Use the gap between simulated and human behavior as a reported finding, not a footnote.
- Document oracle parameterization fully. Claims about convergence speed must specify oracle configuration.

**Phase mapping:** Oracle Agent design. Must be addressed in the first working prototype. Retroactively adding realistic noise after the ablation experiments have run invalidates all simulation results.

---

### Pitfall 3: Soft Assignments Are Not Calibrated — Confidence Does Not Match Reality

**What goes wrong:**
The system produces per-point probability distributions over K clusters (soft assignments). These distributions are uncalibrated: a point assigned with 0.9 confidence to cluster A is not actually a 90% likely member of cluster A. Deep clustering methods are specifically known to be severely overconfident — state-of-the-art methods (SCAN, SPICE) exhibit worse calibration than supervised classifiers.

**Why it happens:**
Soft assignments in clustering are often produced by softmax over embedding distances or pseudo-label propagation. Both methods have no calibration guarantee. Pseudo-labeling in particular creates a compounding overconfidence feedback loop: overconfident pseudo-labels train a model that produces even more overconfident outputs.

The oracle, when asked about boundary points, is shown high-confidence assignments and trusts them, even though they are miscalibrated. The system then uses these assignments to decide which points to surface for clarification — but high confidence is not a reliable signal for "this point is correctly placed."

**Consequences:**
- The system shows the oracle the wrong points (high-confidence wrong assignments instead of low-confidence boundary points).
- `f_uncertainty` is broken: it selects points to surface based on a confidence signal that is systematically overconfident.
- Calibration evaluation (reliability diagrams, ECE) will reveal the problem, but only if it is planned.
- Pairwise validation ("should X and Y be together?") will surface contradictions with stated assignments.

**Warning signs:**
- Soft assignments are produced by softmax without temperature calibration.
- Reliability diagram has not been considered in the evaluation plan.
- `f_uncertainty` selects points purely by entropy of the soft assignment vector.
- No held-out pairwise validation set is maintained.

**Prevention:**
- Apply temperature scaling post-hoc to soft assignments as a standard calibration step. This is low-effort and empirically effective.
- Include a reliability diagram (or at minimum expected calibration error, ECE) as a reported metric.
- `f_uncertainty` should combine calibrated entropy with the oracle's own stated uncertainty signals, not rely on raw model confidence alone.
- Maintain a pairwise validation set: 50-100 point pairs where oracle has stated membership judgments. Track whether soft assignment ordering is consistent with oracle judgments.

**Phase mapping:** Initial clustering implementation. Must be part of the first evaluation loop, not added later.

---

### Pitfall 4: State Management Complexity Kills Iteration Speed

**What goes wrong:**
The system needs to maintain coherent state across turns: current clustering, oracle preference history, contradiction log, soft assignments, hierarchy, and the `f_next_best_step` decision context. As turns accumulate, the state object becomes large and tangled. Updating one part of the state (e.g., merging two clusters after oracle feedback) requires cascading updates across soft assignments, hierarchy, descriptions, and the uncertainty surface. Bugs are introduced. Sessions cannot be reproduced. Ablation experiments fail because state handling differs between conditions.

**Why it happens:**
Multi-turn LLM agents lose context progressively. Research shows a 39% average performance drop in multi-turn vs. single-turn settings, with accuracy falling to ~14% for GPT-4o in complex multi-turn scenarios. State stored only in the conversation context window will silently degrade after 20+ turns.

**Consequences:**
- Late turns produce inconsistent state (contradictory cluster descriptions, stale soft assignments).
- Ablation experiments cannot be run reproducibly.
- The human study cannot be debugged when participants encounter unexpected behavior.
- The `latest intent wins` policy becomes untestable because the contradiction log is unreliable.

**Warning signs:**
- State is passed as raw conversation history rather than a structured object.
- No serialization / deserialization of session state.
- Turn 15+ behavior has not been tested.
- No session replay capability.

**Prevention:**
- Define a typed state schema from day one: `ClusteringState` containing current partition, soft assignments, hierarchy, oracle preference log, contradiction log, and turn history. This is a data structure, not a prompt.
- The Clustering Agent reads from and writes to this state object explicitly at each turn. It does not reconstruct state from conversation history.
- Serialize state to disk after each turn. This gives session replay, enables ablation reproducibility, and surfaces state bugs early.
- Test state integrity at turn 20, 30, 50 with a synthetic oracle before any human study.
- Keep the context window injection small: inject a structured state summary (under 500 tokens) rather than the full conversation history.

**Phase mapping:** Core architecture. Must be established before implementing `f_next_best_step` or `f_next_state`.

---

### Pitfall 5: Contradiction / Preference-Drift Handling Is "Latest Wins" but Never Validated

**What goes wrong:**
The project specifies "latest oracle intent wins on contradictions." This policy is reasonable but requires active tracking and validation to be meaningful. The common failure mode is: the system claims to implement this policy but the underlying state update propagates the latest constraint only partially — old soft assignments, old cluster descriptions, or old hierarchy nodes that conflict with the latest intent are not updated. The system appears to handle contradictions but silently accumulates stale state.

A second failure mode: the system logs "preference drift" as a signal but never uses it. If the oracle's evolving preferences are only stored in a log but never influence `f_next_best_step` (e.g., to ask a clarifying question when drift is detected), then tracking drift adds complexity without benefit.

**Why it happens:**
Preference drift is under-studied: "No IDM (Interactive Decision Making) technique has been explicitly designed to handle preference drift." The default implementation treats all oracle feedback as independent constraints, missing the sequential structure.

**Consequences:**
- Contradiction handling claims cannot be verified.
- The convergence story is false: the system may appear to converge while actually accumulating conflicting constraints.
- The evaluation metric for "contradiction rate" becomes meaningless.

**Warning signs:**
- Contradiction detection is a substring match on the preference log, not a semantic comparison.
- When a contradiction is resolved, only the clustering state is updated, not soft assignments or descriptions.
- Preference drift is logged but not read by any downstream function.

**Prevention:**
- After each state update, run a consistency check: verify that all currently active constraints are satisfiable with the current partition. If not, trigger the contradiction-resolution path explicitly.
- Implement contradiction injection tests: manually force a session where the oracle says "put X in cluster A" and later "put X in cluster B." Verify that the final state, soft assignments, and descriptions all reflect the second instruction.
- Make preference drift an active signal: if drift exceeds a threshold, `f_next_best_step` should ask a clarifying question ("Earlier you said X. Now you seem to prefer Y. Should I treat the earlier constraint as void?"). This also makes drift explicitly measurable.

**Phase mapping:** `f_next_state` and contradiction-handling implementation phase.

---

## Moderate Pitfalls

Mistakes that reduce credibility or require significant rework without invalidating the entire project.

---

### Pitfall 6: The Human Study Is Designed After the System Is Built (Validity Threat)

**What goes wrong:**
The system is built and LLM ablations run. Then, late, a 5-10 person human study is designed to "validate." At this point the study design is constrained by the system's current behavior rather than by methodological requirements. Key issues:
- No within-subject comparison (each person only sees one condition, preventing comparison).
- No counterbalancing of condition order.
- No pre-study measurement of participants' prior familiarity with the dataset.
- Consent forms drafted in a hurry, missing elements required for academic publication.
- Session duration not piloted: participants quit before finishing.

**Prevention:**
- Write the human study protocol before implementing the full system. The protocol specifies: study design (within-subject recommended for N=5-10), conditions, counterbalancing, session duration limit (30-45 minutes max), consent procedures, and how data will be aggregated.
- Include a 2-person pilot run during the alpha testing phase, not after.
- Measure: time-to-complete, turns-to-satisfaction, self-reported cognitive load (NASA-TLX or simplified Likert), and pairwise agreement with LLM oracle on the same tasks.
- Treat the human study as primary validation, not a checkbox.

**Phase mapping:** Study protocol written in the evaluation design phase. Pilot run in alpha testing.

---

### Pitfall 7: Cognitive Load Is Discovered at Study Time, Not Budgeted at Design Time

**What goes wrong:**
The system shows the oracle: current cluster descriptions (all K clusters), the point being discussed, its embedding neighbors, current soft assignment, and a proposed action. This is 4-6 distinct pieces of information per turn. Cognitive load research shows that working memory can hold ~4 chunks simultaneously. By turn 10, the oracle has accumulated partial memories of previous decisions that interfere with current decisions. Participants report confusion and fatigue. Measured turns-to-convergence is inflated because the oracle makes errors driven by overload.

**Why it happens:**
Information display is designed to be informative ("show everything useful") rather than to minimize cognitive load. The research constraint ("budgeted from turn one") is stated but not operationalized into concrete per-turn display limits.

**Prevention:**
- Define a maximum information budget per turn before any UI/prompt design: recommend no more than 3 information elements per turn (e.g., one cluster description + one point + one yes/no question).
- Use progressive disclosure: offer summary first, details on request.
- `f_output` must include a cognitive-load cost model. Each additional element shown has an estimated cost. The total per-turn cost has a hard cap.
- Measure self-reported cognitive load (even a 1-item "how difficult was this turn?" slider) from the first human pilot.

**Phase mapping:** `f_output` design, before any human-facing prompts are finalized.

---

### Pitfall 8: Generalization Is Deferred Until the End and Has No Evaluation Plan

**What goes wrong:**
The requirement to "codify oracle preferences into a function for new items" is listed but has no evaluation design. The frozen held-out subset exists, but there is no specification of what success looks like for new-item assignment. Common failure: the generalization function is tested on the held-out set but the oracle was implicitly consulted on some of those items during development. The held-out set is contaminated.

**Prevention:**
- Lock the held-out subset before any development begins. Write its file path and hash to a file that is never modified.
- Define the generalization metric before building the generalization function: "Given the oracle's preference function derived from the interaction session, what fraction of held-out items are assigned to the same cluster the oracle would choose on first presentation?"
- The Oracle Agent must be forbidden from seeing held-out items during simulation runs. Enforce this programmatically.

**Phase mapping:** Dataset preparation (before coding). Evaluation planning (before interaction implementation).

---

### Pitfall 9: Stopping Signal Is Implicit or Circular

**What goes wrong:**
The stopping signal is described as "oracle satisfaction, diminishing returns, or turn budget." Without an explicit operationalization, the system either:
- Never stops (runs until the oracle stops responding).
- Stops when the oracle explicitly says "I'm done," which is a self-report with high variance between personas.
- Uses diminishing returns on clustering quality, which is the non-interactive metric from Pitfall 1.

A circular stopping signal: the system stops when the Judge Agent's `f_eval` score exceeds a threshold, but `f_eval` uses oracle satisfaction as input, so the oracle is being asked to both produce feedback and determine when to stop producing feedback.

**Prevention:**
- Operationalize the three stopping criteria independently and test each:
  - Turn budget: hard cap (e.g., 20 turns). Always active.
  - Diminishing returns: defined as "fewer than X points changed cluster assignment in the last N turns" — purely based on state change, not oracle evaluation.
  - Oracle satisfaction signal: a structured end-turn response the oracle is prompted to produce ("DONE" vs "CONTINUE") at each turn, not inferred from natural language.
- The Judge Agent should not query the oracle to decide whether to stop. It reads the state change signal and the turn count.

**Phase mapping:** `f_next_best_step` and Judge Agent implementation.

---

## Minor Pitfalls

Mistakes that reduce polish or efficiency but are recoverable.

---

### Pitfall 10: Ablation Conditions Are Not Fully Crossed

**What goes wrong:**
Three to five interaction strategies are ablated, but conditions share oracle instances. If oracle persona A is used with strategy 1 and oracle persona B with strategy 2, any difference in outcome is confounded with oracle persona.

**Prevention:**
Run each interaction strategy with each oracle persona configuration. Minimum: 3 strategies × 3 oracle personas × 3 dataset seeds = 27 runs. This is feasible with LLM oracles. Report results as strategy × persona interaction, not strategy main effect alone.

**Phase mapping:** Ablation experiment design.

---

### Pitfall 11: Cluster Descriptions Are Generated Once and Become Stale

**What goes wrong:**
Initial cluster descriptions are generated before the oracle has seen the data. As oracle feedback reshapes the clusters, descriptions are not updated. The oracle is then shown descriptions that no longer match the cluster contents. This increases cognitive load and introduces confusion about whether the cluster description or the cluster membership takes precedence.

**Prevention:**
Cluster descriptions are outputs of `f_output` and must be regenerated (or at minimum flagged for review) after any state change that affects cluster membership. This is a state dependency that must be explicit in the state schema.

**Phase mapping:** `f_output` and `f_next_state` implementation.

---

### Pitfall 12: Confidence Intervals Are Added Post-Hoc

**What goes wrong:**
Claims are made first; confidence intervals are added to satisfy the project requirement ("at least one defensible quantified claim with confidence intervals"). This produces narrow intervals on selected metrics and wide intervals on the primary metric — the opposite of what a legitimate claim requires.

**Prevention:**
Decide the primary quantified claim before running experiments: e.g., "systems with `f_next_best_step` reach oracle satisfaction in fewer turns than random-action baseline (95% CI)." Run enough simulation trials to produce a meaningful CI. Bootstrap resampling across oracle seeds is appropriate for LLM oracle runs.

**Phase mapping:** Evaluation design, before any experiment runs.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Evaluation design | Pitfall 1: no baseline condition | Define no-dialogue and random-interaction baselines first |
| Oracle Agent design | Pitfall 2: over-consistent oracle | Add explicit noise/drift/sycophancy-resistance parameters from the start |
| Initial clustering | Pitfall 3: uncalibrated soft assignments | Apply temperature scaling; add reliability diagram to eval suite |
| State schema design | Pitfall 4: state management collapse | Define typed `ClusteringState` object; serialize after every turn |
| Contradiction handling | Pitfall 5: partial state updates | Write contradiction injection tests; active drift detection |
| Human study protocol | Pitfall 6: late study design | Write protocol before building system; pilot with 2 people in alpha |
| Per-turn display design | Pitfall 7: cognitive overload | Hard cap on information elements per turn; measure NASA-TLX from day 1 |
| Dataset preparation | Pitfall 8: held-out set contamination | Lock and hash held-out split before any code is written |
| Stopping criteria | Pitfall 9: circular/implicit stopping | Three operationalized stopping criteria, each independently testable |
| Ablation experiment | Pitfall 10: confounded conditions | Cross oracle personas × interaction strategies |
| Cluster description updates | Pitfall 11: stale descriptions | Track description freshness in state; auto-flag on cluster update |
| Quantitative claims | Pitfall 12: post-hoc CI | Specify primary claim and required sample size before running experiments |

---

## What Previous Interactive Clustering Papers Got Wrong

Based on the scoping review (Springer, 2020, 50 primary studies) and the comprehensive ACM review (2020, 105 papers):

**The three most consistent gaps documented in the literature:**
1. **Evaluation of expert supervision** — most papers do not evaluate whether and how human feedback actually improved outcomes versus a non-interactive baseline.
2. **Evaluation of expert effort** — turns, time, cognitive load, and decision complexity are almost never measured. "The algorithm converges" is reported without measuring what it cost the human.
3. **Meaningfully involving human experts** — most studies use synthetic feedback, small-scale user studies with non-expert participants, or skip human validation entirely. Oracle behavior is modeled as noise-free constraints, not realistic human preferences.

**What this means for this project:**
All three gaps are explicitly identified in the PROJECT.md requirements and constraints. The risk is not that the project misses them in its stated goals — it is that implementation pressure leads to deferring their measurement to "evaluation," at which point the system design no longer supports them. The prevention strategy is to make all three gaps first-class engineering requirements, not research questions to be answered later.

---

## Sources

- Interactive Clustering scoping review (Springer, 2020): https://link.springer.com/article/10.1007/s10462-020-09913-7
- Interactive Clustering comprehensive review (ACM, 2020, 105 papers): https://dl.acm.org/doi/fullHtml/10.1145/3340960
- SycEval: LLM Sycophancy Evaluation (2025): https://arxiv.org/html/2502.08177v2
- The Challenge of Using LLMs to Simulate Human Behavior (causal inference): https://arxiv.org/html/2312.15524v1
- Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation: https://arxiv.org/html/2509.03736v1
- Towards Calibrated Deep Clustering Network: https://arxiv.org/html/2403.02998v2
- Limitations of Current Evaluation Practices for Conversational Recommender Systems: https://arxiv.org/html/2510.05624
- Reward Hacking in RLHF (Lilian Weng): https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- LLMs Get Lost In Multi-Turn Conversation: https://arxiv.org/html/2505.06120v1
- Why Do Multi-Agent LLM Systems Fail?: https://arxiv.org/html/2503.13657v3
- Handling concept drift in preference learning for interactive systems: https://www.researchgate.net/publication/228967853_Handling_concept_drift_in_preference_learning_for_interactive
- HITL Machine Learning state of the art (Springer, 2022): https://link.springer.com/article/10.1007/s10462-022-10246-w
- Stable reliability diagrams for probabilistic classifiers (PNAS): https://www.pnas.org/doi/10.1073/pnas.2016191118
- Dial-In LLM: Human-Aligned Dialogue Intent Clustering with LLM-in-the-loop: https://arxiv.org/html/2412.09049v1

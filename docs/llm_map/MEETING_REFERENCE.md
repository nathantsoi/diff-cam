# CS 370 Independent Study — First-Meeting Reference

**Meeting:** Wednesday, September 2, 2026  
**Working goal:** Understand the existing DiffCAM work, locate the current A/B critique loop, clarify the paper's intended claims, and begin framing the literature review.

## 30-second project summary

DiffCAM uses continuous signed-distance-field geometry and analytic gradients to optimize CNC toolpaths. The independent study adds an LLM-based meta-optimization loop: a human compares two rendered toolpaths, selects A/B/Tie, and provides a free-form critique. The alternatives shown to the human can change the human's expectations and priorities; the choice and critique are therefore treated as observations of an evolving latent human state. A persistent agent uses this history to modify DiffCAM's losses, weights, schedules, search process, or code, then generates the next comparison. The intended contribution is alignment of the **optimization process**, not merely alignment of a learned policy.

## One-picture mental model

```text
Current agent belief / living state (b_t)
                    ↓
Choose optimizer change and generate A/B candidates (a_t)
                    ↓
Human sees the resulting toolpaths
                    ↓
Human priorities/expectations change (p_t → p_t+1)
                    ↓
Human selects A/B/Tie and writes a critique (o_t+1)
                    ↓
Digest feedback and update the agent's belief (b_t+1)
                    ↓
Modify DiffCAM and repeat
```

Short version:

> DiffCAM optimizes a toolpath. The proposed LLM loop meta-optimizes how DiffCAM performs that optimization as it learns what the human means by a good machining outcome.

## What the semester appears to involve

1. Review literature on preference learning from free-form feedback.
2. Identify and refine defensible research claims.
3. Design a repeatable experimental protocol supporting those claims.
4. Implement any missing agent, interface, experiment, and data-collection scaffolding.
5. Execute experiments and produce paper-ready data and figures.
6. Progressively draft Related Work, Method, Results, and Future Work.
7. Finish with a documented DiffCAM codebase and a venue-ready robotics paper.

## Vocabulary to use precisely

- **CAM:** Programming how a target part will be manufactured.
- **CNC:** The machine executing the manufacturing instructions.
- **Stock:** The original block of material.
- **Toolpath:** The cutting tool's trajectory.
- **B-Rep:** A boundary/surface representation. Useful for static geometry, but Boolean material-removal operations do not provide useful learning gradients.
- **Voxel representation:** A dense 3D grid. Straightforward but has roughly \(O(N^3)\) memory scaling.
- **SDF/PSDF:** A continuous function giving signed distance to a surface; the zero level set defines the surface. Parametric SDFs represent parts using combinations of primitives.
- **Differentiable simulator:** A simulator that can propagate derivatives from an outcome/loss back to toolpath or policy parameters.
- **Analytic gradient:** Directional information indicating how a parameter should change to reduce a loss, avoiding purely stochastic trial and error.
- **Proxy objective:** A measurable objective used in place of the human's true, incompletely specified goal.
- **Surrogate drift:** Divergence between the soft differentiable training objective and the hard deployable evaluation metric.
- **Meta-optimization:** Modifying the objective, optimizer, schedules, search process, or code that produces the toolpath—not directly selecting every tool motion.
- **Latent human state \(p_t\):** Unobserved priorities, expectations, trust, understanding, or satisfaction.
- **Observation \(o_t\):** Evidence about that state, here an A/B/Tie choice plus a textual critique.
- **Belief state \(b_t\):** The agent's current, history-dependent estimate of the latent human state.
- **Coadaptation:** The human changes in response to agent-generated alternatives while the agent changes in response to human feedback.

## Paper 1: OVPR — Differentiable Simulation for Manufacturing

### Core argument

```text
Manual CAM is expensive and expert-intensive
                 ↓
AI needs a useful machining simulator
                 ↓
B-Reps lack useful gradients; voxels scale poorly
                 ↓
PSDFs provide continuous implicit geometry
                 ↓
Differentiable material removal and collision losses
                 ↓
Gradient-guided, continuous, collision-aware toolpaths
```

### Preliminary baseline versus proposed DiffCAM

The preliminary system used a voxel state, a 3D CNN, discrete tool motions, PPO, and a hand-designed composite reward. Figure 1 shows that this baseline can cut material while allowing the passive tool holder to collide with the stock.

That collision **motivates** a differentiable collision penalty; it does not demonstrate that DiffCAM successfully fixes the problem. Use these phrases carefully:

- **“The proposal hypothesizes…”** for the complete DiffCAM system and its advantages.
- **“The preliminary baseline demonstrates…”** for the voxel/PPO cutting behavior and collision failure.
- Do not say the proposal has already established PPO/SAC superiority, robustness, or physical deployment.

### Four OVPR milestones

1. **Continuous volumetric representation:** Build stock, target, and tools from PSDF primitives.
2. **Differentiable safety:** Separate cutting flutes from the passive holder and formalize collision penalties.
3. **Reward design and gradient flow:** Combine volumetric, surface, normal-consistency, and safety losses and backpropagate to a continuous toolpath.
4. **Evaluation/deployment:** Compare with PPO and SAC; measure time, accuracy, IoU, Chamfer distance, robustness, gradient stability, and eventually physical machining.

### Meeting-ready framing

> DiffCAM removes the need for human oversight for safety constraints that are already formalized, but it cannot identify every relevant constraint or qualitative objective. The coadaptive loop may help expose those missing specifications, with its main intervention occurring in reward and optimizer design.

> DiffCAM reduces trial-and-error in toolpath optimization, but it does not eliminate trial-and-error in specifying the correct objective.

## Paper 2: Coadaptive Value Alignment

### Core distinction

Traditional alignment is largely unidirectional: the human provides demonstrations, rewards, or preferences, and the agent optimizes a fixed objective. Coadaptive Value Alignment instead models a closed loop in which agent actions affect the human's internal state and the human's resulting observations affect the agent.

### POMDP mapping to this project

| CVA concept | Proposed CNC instantiation |
|---|---|
| Task state \(e_t\) | Simulator, experiment, optimizer, and rollout state |
| Human state \(p_t\) | Latent machining priorities, expectations, and satisfaction |
| Observation \(o_t\) | A/B/Tie choice plus critique |
| Belief \(b_t\) | Persistent living-agent state |
| Action \(a_t\) | Experiment choice, optimizer/code modification, and candidates presented |
| Task reward | Geometry, collision avoidance, efficiency, hard metrics |
| Satisfaction/alignment | Agreement with the human's qualitative judgment |

Important correction:

> The critique is not \(p_t\). It is an observation providing evidence about the hidden state \(p_t\).

### What “coadaptive” is intended to mean here

The human does not merely reveal a fixed preference. Seeing A/B alternatives can teach the human about tradeoffs, expose a previously unnoticed failure, recalibrate expectations, or change priorities. The agent controls which alternatives are presented and therefore partly controls the history through which the human's state evolves.

### Evidence in the CVA paper

The Spot-robot case study reports strong positive Spearman correlations between instructed competence and perceived competence for blind observers and teleoperators. This supports one link—robot actions can affect human perception—but does not validate the complete POMDP, belief update, or adaptive policy.

### Ethical concern

**Influential inseparability:** A system capable of helping a person refine preferences can also steer those preferences toward outcomes convenient for the system. Relevant safeguards include randomized A/B order, diverse candidate generation, transparent experiment records, and a complete audit trail.

### Meeting-ready framing

> Are we treating critiques as observations of an evolving latent human state, or simply as instructions for iterative reward design? What evidence and implementation choices will justify calling the loop coadaptive?

## Paper 3: Freeform Preference Learning (FPL)

### What “freeform” means in FPL

FPL generally asks annotators to name natural-language **comparison axes**—for example, speed, smoothness, hygiene, or placement quality—and then select A/B/equivalent separately for each axis. It does not primarily interpret an unrestricted explanatory critique.

Some experiments use predefined axes. The Plate Toast task is the main open-ended-axis setting.

### FPL method

1. Show two full trajectories.
2. Collect one or more natural-language axes \(l_k\) and per-axis pairwise labels \(y_k\).
3. Train a language-conditioned reward model \(r_\phi(\tau \mid l)\).
4. Preserve the axes instead of collapsing them into one scalar.
5. Train a reward-conditioned policy using the axis names and desired scores.
6. Optionally repeat with new policy rollouts and new preferences.

### What FPL reports

- Four real-world and two simulated manipulation tasks.
- Twenty real-world rollouts per method and three simulation seeds.
- Average real-world success of 0.75 for FPL versus 0.37 for the next-best reported baseline: a 38-percentage-point improvement.
- Compositionality: combining attributes not jointly demonstrated, such as a fast behavior with a target placement seen only in slow demonstrations.
- Test-time steerability: changing the requested reward dimensions without retraining the policy.
- Qualitatively denser credit assignment on long-horizon tasks.
- Preference axes shift from coarse completion criteria to finer quality criteria as policies improve.
- Multi-axis annotation amortizes the cost of watching a pair of videos and is reported as faster per label.

### FPL versus this project

| FPL | Proposed DiffCAM/CVA project |
|---|---|
| Language names a scoring axis | Language explains a preference and potentially changing priorities |
| A/B label is supplied separately per axis | One A/B/Tie choice is accompanied by an unrestricted critique |
| Learns \(r_\phi(\tau\mid l)\) | LLM maintains a stateful hypothesis about latent human intent |
| Aligns/trains a reward-conditioned policy | Meta-optimizes DiffCAM's objective and optimization process |
| Preference buffer is reward-model training data | Feedback history is intended to be part of the agent's belief state |
| Does not explicitly model how shown options change the human | Candidate presentation is intended to cause \(p_t \rightarrow p_{t+1}\) |
| Changing axes are observed as a curriculum | Preference evolution and contradiction are intended modeling targets |

### Best literature framing

> FPL demonstrates that preserving human-specified, natural-language preference dimensions produces less ambiguous supervision than a single overall binary preference and can support dense reward learning, compositionality, and policy steerability. Our work studies a complementary but distinct setting: unrestricted critiques are treated as sequential observations of an evolving latent human state, and an LLM-based meta-optimizer uses those observations to modify the objective and optimization dynamics of a differentiable simulator rather than directly learning a preference-conditioned policy.

Shorter version:

> FPL learns separate rewards for human-named preference axes. We want to interpret open-ended rationales over time, maintain a belief about evolving human intent, and use that belief to modify the optimizer itself.

Even shorter:

> FPL treats language as a coordinate system for reward learning; we treat language as evidence about a changing human objective and as input to a meta-optimizer.

## Current Text-Based Critiques / LLM-MAP draft

### Proposed components

1. Parameterizable optimizer knobs.
2. Standardized visual rollout generation.
3. Web-based A/B/Tie preference interface with text critique.
4. Digest aggregating choices, manipulated knobs, and critiques.
5. Persistent living-agent state containing preference themes and hypotheses.
6. Active selection of subsequent sweeps or structural modifications.

The draft reports a **+0.181 mean hard-Dice gain** associated with surrogate-sharpness annealing and hard-metric checkpoint selection. Treat this as a reported draft result until the repository, experimental logs, run counts, and causal attribution are verified.

Useful phrasing:

> The LLM does not directly control the CNC trajectory. DiffCAM produces the trajectory; the LLM operates one level above it, modifying how the optimizer searches and what it values.

## Candidate contribution claims — hypotheses, not yet finalized

These are possible directions to test, not statements currently supported by complete evidence:

1. Free-form critiques enable more effective optimizer adaptation than binary preferences alone.
2. A persistent, history-aware agent handles evolving or contradictory preferences better than a memoryless critique-to-parameter mapping.
3. LLM-driven meta-optimization can discover useful changes to optimizer dynamics, not only tune fixed reward weights.
4. Active candidate generation elicits more informative feedback than passive or random comparison selection.
5. Human-guided meta-optimization improves hard deployable CNC outcomes while retaining geometric safety and efficiency.

## Questions for Dr. Tsoi

### Highest priority: scope and status

1. Which OVPR milestones are currently implemented in the `autoresearch` branch?
2. What parts of the A/B critique loop already work end-to-end?
3. Where is the web interface you demonstrated, and is it included in this branch?
4. What work has already been completed versus what is expected from me this semester?
5. Is the intended deliverable one CNC-only paper, or is the current draft still meant to include multiple domains?
6. Which venue and submission window should guide scope and evaluation?

### Technical architecture

7. Does DiffCAM optimize toolpath coordinates directly, parameters of a policy, or both?
8. Which loss terms, weights, hyperparameters, and schedules are currently exposed to the agent?
9. What exactly are soft Dice and hard Dice in this implementation?
10. Where does the differentiable surrogate enter, and how is its sharpness parameterized?
11. What may the LLM modify: a bounded configuration, experiment files, arbitrary source code, or all three?
12. How are unsafe or invalid agent modifications prevented and recovered?
13. What is the unit of an A/B comparison: final geometry, complete toolpath animation, training curve, or a bundle of outputs?

### Human-state and feedback model

14. What dimensions of latent human state are we claiming to model: technical priorities, expectations, satisfaction, expertise, or something else?
15. Is the living-agent state intended as a formal approximation of the CVA belief state \(b_t\)?
16. How does the system distinguish a new critique from an inferred change in underlying preference?
17. How should contradictions be handled: preference drift, context dependence, annotator noise, or unresolved uncertainty?
18. Is feedback personalized to one individual, pooled across people, or both?
19. Does the agent intentionally select candidates for information gain, preference shaping, performance improvement, or some combination?
20. What evidence would be sufficient to call the loop coadaptive rather than iterative reward engineering?

### Claims and existing evidence

21. Was surrogate-sharpness annealing autonomously discovered by the agent, suggested by a researcher, or found through prior manual experimentation?
22. How many runs produced the reported +0.181 mean hard-Dice improvement?
23. What baseline and variance does that number use?
24. Has DiffCAM been compared against PPO or SAC yet?
25. Which results are stable enough to frame as claims now, and which remain exploratory?
26. Is physical machining within this semester's scope, or will evaluation remain simulated?

### Experimental design

27. What is the primary comparison: no feedback, binary-only feedback, FPL-style axes, free-form critiques, or a memoryless LLM baseline?
28. What ablations are expected: critique text removed, history removed, active selection removed, code modification disabled, or fixed knobs only?
29. How will we measure preference alignment independently of the metric being optimized?
30. How many users, feedback rounds, random seeds, and geometries are feasible?
31. Should A/B presentation order be randomized?
32. What data format should preserve candidates, configurations, feedback, belief-state updates, code diffs, metrics, and provenance?
33. What constitutes a repeatable experiment when an LLM may make different changes across runs?

### Literature review and writing

34. Is FPL the primary closest related work, or are there specific RLTF/TAMER/comparative-language papers you want emphasized first?
35. Should novelty be framed primarily as free-form critique, meta-optimization, coadaptation, active exploration, or their combination?
36. Should the Related Work section distinguish preference **formation** from preference **revelation**?
37. How strongly should the paper claim recovery of “latent human intent” versus adaptation to observed critiques?

## Important conceptual ambiguities to resolve

### Preference formation versus preference revelation

Changing feedback can mean several things:

1. The human's underlying values genuinely changed.
2. An existing value became relevant only after a new failure appeared.
3. The human learned enough to articulate a preference they already held.
4. The interface or presented alternatives anchored the human's judgment.

All four produce temporal changes in feedback, but only some are genuine value drift. The paper should define what counts as a change in \(p_t\).

### Coadaptation versus iterative reward engineering

A loop of “human complains → LLM changes a weight → repeat” is not automatically a validated instance of Coadaptive Value Alignment. A stronger instantiation should include:

- persistent history;
- an explicit distinction between observations and inferred state;
- candidate-to-feedback provenance;
- representation of uncertainty and conflict;
- state-dependent selection of the next comparison;
- comparison with a memoryless baseline;
- evaluation under preference drift or contradiction.

### Discovery versus tuning

Changing values inside a fixed set of knobs is hyperparameter/reward tuning. Introducing a new loss, schedule, metric, or optimization mechanism is a stronger form of structural meta-optimization. The experimental claims should distinguish these levels.

## Codebase inspection checklist for later

When reviewing `https://github.com/nathantsoi/diff-cam/tree/autoresearch`, locate and trace:

- A/B/Tie web interface and routes;
- free-form critique input;
- candidate queue and rollout pairing;
- randomized A/B presentation, if any;
- feedback database/schema and history;
- digest generator;
- living-agent state file;
- experiment driver/enqueue logic;
- parameter sweep definitions;
- renderer and generated artifacts;
- link from feedback to the next experiment;
- LLM prompts, tools, permissions, and writable files;
- code-diff validation and rollback;
- soft/hard metrics and checkpoint selection;
- surrogate-sharpness/annealing implementation;
- experiment logs, seeds, configurations, and result tables;
- any existing paper outline, `autoresearch.md`, or analysis scripts.

## Evidence-language guardrails

Use language that matches the maturity of the evidence:

- **“The OVPR proposal hypothesizes…”**
- **“The preliminary PPO baseline demonstrates…”**
- **“The CVA paper conceptualizes…”**
- **“The Spot study supports the action-to-perception link…”**
- **“FPL reports/demonstrates in its tested tasks…”**
- **“The current LLM-MAP draft proposes…”**
- **“The draft reports +0.181 mean hard Dice, pending verification of the underlying runs…”**
- **“Our possible contribution is…”** until claims and protocols are finalized.

## Source materials in this workspace

- `Brief Outline Fall CS 370.docx`
- `Relevant Papers/OVPR_New_Directions___Differentiable_Simulation_for_Manufacturing.pdf`
- `Relevant Papers/Coadaptive_Value_Alignment_Tsoi_2026.pdf`
- `Relevant Papers/Freeform_Preference_Learning_for_Robotic_Manipulation_Torne.pdf`
- `Text-Based Critques.pdf`
- Repository to inspect later: `https://github.com/nathantsoi/diff-cam/tree/autoresearch`

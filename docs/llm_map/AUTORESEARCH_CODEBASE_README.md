# Personal README: `diff-cam` Autoresearch Branch

> First-pass orientation to Nathan Tsoi's [`autoresearch` branch](https://github.com/nathantsoi/diff-cam/tree/autoresearch), reviewed at commit `41b7355` on 2026-09-01. This is a reading guide to the current implementation, not the repository's official documentation.

## One-sentence description

The branch combines a differentiable CNC toolpath optimizer with an asynchronous A/B web interface: a human compares trajectories and explains their choice, then an external coding agent interprets that feedback and changes subsequent experiments, objective weights, initialization strategies, or code.

## The system at a glance

```text
Generate two trajectories that differ in one selected setting
        |
        v
Human views synchronized A/B trajectories
        |
        v
Human chooses A, B, or tie and writes a free-form critique
        |
        v
Choice and critique are stored in pairwise.json
        |
        v
A simple digest counts results and collects recent comments
        |
        v
External coding agent updates its written preference understanding
and selects or implements the next experiment
        |
        v
GradMill directly optimizes another trajectory through the simulator
```

The feedback pipeline exists end to end. The interpretation and objective-update step is currently performed by the external coding agent rather than a formal learned human-state model.

## Where the major pieces live

### Core optimizer and simulator

- [`algorithms/train_csg.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/algorithms/train_csg.py) is the main GradMill training program. It initializes a toolpath, runs differentiable simulation, sends trajectory gradients to Adam, evaluates soft and hard carves, selects the best checkpoint, and saves metrics and artifacts.
- [`simulator/csg_simulator.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/simulator/csg_simulator.py) implements the differentiable and sharp CSG carving operations, motion constraints, geometry loss, trajectory regularizers, and safety/efficiency surrogates.
- [`simulator/csg_metrics.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/simulator/csg_metrics.py) contains geometric residual and gouge measurements.
- [`eval/eval_csg.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/eval/eval_csg.py) supplies shared evaluation metrics such as Dice, ASD, and HD95.

### A/B preference interface

- [`autoresearch/tasks/train_csg/web/compare.html`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/autoresearch/tasks/train_csg/web/compare.html) is the web interface shown to the human. It presents synchronized 3D trajectories, A/B/tie controls, and a free-form “Why?” field. It also displays the experimental dimension, A/B magnitudes, scenario, and a preference digest.
- [`scripts/serve_web_https.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/scripts/serve_web_https.py) serves the interface and implements the pairwise and per-run feedback endpoints.
- [`scripts/enqueue_pair.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/scripts/enqueue_pair.py) adds a completed pair of runs to the comparison queue.
- [`autoresearch/tasks/train_csg/sweep_pref_pair.sh`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/autoresearch/tasks/train_csg/sweep_pref_pair.sh) launches two runs while varying one selected objective setting and then enqueues them as a pair.

### Feedback interpretation and experiment loop

- [`scripts/pref_lib.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/scripts/pref_lib.py) groups answered comparisons by dimension, counts A/B/tie responses, and collects written reasons.
- [`scripts/pref_digest.py`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/scripts/pref_digest.py) prints that summary for the coding agent.
- [`autoresearch/tasks/train_csg/autoresearch.md`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/autoresearch/tasks/train_csg/autoresearch.md) is the coding agent's operating protocol. It instructs the agent to keep generating comparisons, read new critiques, update its interpretation, and modify experiments or code.
- [`autoresearch/tasks/train_csg/pref_objective_plan.md`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/autoresearch/tasks/train_csg/pref_objective_plan.md) contains the current written preference understanding, experimental plan, summarized feedback, and outcomes of preference-inspired interventions.
- [`autoresearch/tasks/train_csg/web/index.html`](https://github.com/nathantsoi/diff-cam/blob/autoresearch/autoresearch/tasks/train_csg/web/index.html) is the broader run dashboard. It has a separate 1–7 star rating and free-text feedback mechanism for individual runs.

## What GradMill actually learns

GradMill does **not** learn a neural policy. For a trajectory with 128 tool positions, it directly optimizes approximately 127 three-dimensional displacement vectors. Those displacements are the learned artifact.

During an iteration:

1. The current displacements are loaded into the Taichi simulator.
2. The simulator starts with a block of stock.
3. It executes each movement subject to feed/rapid speed limits and a vertical floor.
4. Each swept cutter segment subtracts material from the stock.
5. The final stock is compared with the target part.
6. Geometry, motion-quality, efficiency, and safety terms are added to the loss.
7. Taichi differentiates through the full sequence of cuts.
8. Adam updates every displacement vector.

Structured initial trajectories—contours, helices, multi-depth paths, cavity-aware paths, and related variants—are important because the optimization problem is highly nonconvex. The initialization places Adam in a useful region of the search space.

## Soft training versus hard evaluation

Training uses a smooth approximation to Boolean CSG subtraction so that small trajectory changes produce useful gradients. Evaluation also runs a sharp, non-differentiable carve intended to represent the deployable result.

This creates the recurring **soft/hard gap**: smooth training loss can continue improving while the sharp carve becomes worse. The branch addresses this principally through:

- **Smoothness (`k`) annealing:** broad, smooth gradients early and sharper hard-carve tracking later.
- **Best-on-hard checkpointing:** periodically evaluate the sharp carve and preserve the best deployable checkpoint rather than automatically using the final iteration.

The checkpoint score is dominated by hard Dice, with smaller penalties for air time, total time, and predicted breakage risk.

## Current objective vocabulary

The main human-relevant concepts already represented in code are:

| Concern | Relevant code-level objective or mechanism |
|---|---|
| Unwanted material remains | `w_residual`; near-surface `w_annulus` |
| Cutter removes desired material | `w_gouge`; direct-position `w_tool_gouge` |
| Holder collides with stock | holder penetration barrier, vertical floor, hard collision diagnostics |
| Tool wanders away from the surface | `w_prox`, `w_traj_prox` |
| Excessive air cutting | `w_air`, `w_air_time`, `w_air_late` |
| Useless trailing movement | `w_len`, late-air term |
| Abrupt direction or speed changes | `w_jerk` |
| Inconsistent step lengths/feed | `w_step` |
| Long machining time | `w_time` |
| Tool-break risk | differentiable force/risk surrogate via `w_break` |
| Surface finish, scallops, or inadequate pass overlap | no adequate direct objective yet |

The geometry core balances two errors:

- **Residual:** stock remains where the target should be empty.
- **Gouge:** stock is missing where the target should remain solid.

The newer direct tool-position gouge barrier exists because a loss defined only on smooth stock occupancy can look satisfied even when the sharp carve still gouges.

## How human feedback currently affects future runs

Pairwise records contain the run IDs, comparison dimension, A/B magnitudes, scenario, answer, written note, and timestamps. The digest reduces them to A/B/tie counts per dimension plus recent notes.

`train_csg.py` reads and records the pairwise summary, but explicitly treats it as a **steering signal, not a loss term**. The actual bridge is:

1. The coding agent reads the digest and raw critique themes.
2. It updates the prose “current preference understanding.”
3. It chooses another comparison or edits a weight, initialization, metric, or loss implementation.
4. Hard-Dice experiments determine whether the intervention is retained.

The separate star-rating system can optionally warm-start a new run from the highest-rated prior trajectory matching the target shape and trajectory length. That is an automated feedback effect, but it is separate from learning from pairwise critiques.

## What the feedback has surfaced so far

The committed preference summary emphasizes:

- Better contour and surface following
- Less end-of-path air cutting
- Poor or jagged surface finish
- Insufficient overlap between cuts
- Residual material and gouging tradeoffs

Several preference-inspired parameter and initialization changes were neutral or negative under hard Dice. This is informative: the human appears to notice trajectory qualities that hard Dice does not measure. Surface finish and overlap are especially important examples of concerns for which the present objective vocabulary is incomplete.

## What exists versus what remains conceptual

### Implemented

- Differentiable direct trajectory optimization
- Soft training and sharp deployable evaluation
- Many geometry, efficiency, motion-quality, and simulated-safety terms
- A/B/tie comparisons with free-form critique
- Persistent local feedback storage and a web digest
- A written, accumulating preference interpretation
- Agent-directed experiments and permission to modify the objective/code
- Optional warm-starting from highly rated trajectories

### Not yet implemented formally

- A learned representation of the human's latent state
- A model of how viewing a comparison changes that state
- A distinction between preference change, context, uncertainty, and noisy answers
- Automatic semantic interpretation of critiques inside the repository
- A principled uncertainty or information-gain query strategy
- A validated procedure for turning a new verbal concern into a faithful metric
- Direct lineage from a particular critique to a belief update, code change, and outcome
- Real-machine human-oversight or safety validation

## Brief FPL versus coadaptive interpretation

At present, the loop is best described as **FPL-inspired, LLM-mediated objective engineering**: language helps the agent identify relevant preference dimensions and propose better objectives.

To make the coadaptive distinction concrete, a modest next step would be to maintain a time-indexed belief state containing current features, candidate features, estimated importance, uncertainty, and supporting evidence. The system would preserve what the human saw before each response, validate newly proposed features with diagnostic comparisons, and detect when recent preferences conflict with earlier ones rather than pooling all feedback together.

Feature expansion alone is not enough to establish coadaptation; the research would also need to test whether interaction changes the human's priorities and whether a state-aware model adapts better than a static FPL baseline.

## Important first-pass caveats

- The digest aggregates “prefers A” versus “prefers B,” rather than the actual selected setting. A/B order is not randomized, so presentation-side effects and inconsistent ordering could distort the summary.
- The UI reveals the manipulated dimension and both magnitudes. This is useful diagnostically but may anchor the human's critique; a blinded evaluation mode would provide stronger evidence.
- Pair selection is heuristic—underexplored dimensions, weak signals, and recurring themes—not formal active learning or expected information gain.
- The instructions encourage continuously producing comparisons even when many remain unanswered. The current code therefore does not by itself demonstrate sampling efficiency.
- Hard Dice is the advancement gate even when feedback concerns qualities it cannot measure. It should be clarified whether Dice is the primary goal, a safety/deployability constraint, or one component of a multi-objective evaluation.
- The command-line enqueuer and web server modify the same JSON store with different process-local locks. Atomic file replacement prevents corruption but may not prevent a simultaneous enqueue and answer from overwriting one another.
- Raw `pairwise.json`, star ratings, and run artifacts are not committed. The repository's prose summaries cannot be independently reconstructed without access to the live data.
- The pair-generation shell script includes an environment-specific absolute path and may require cleanup before use elsewhere.
- Reported improvements should be checked against the underlying experiment records; the paper draft and branch comments do not appear completely synchronized.

## Questions to keep in view

1. Is the Markdown preference summary intended to be the human-state representation, or only a temporary scaffold?
2. What is the precise sampling-efficiency claim, comparison budget, and baseline?
3. Is hard Dice a primary objective, a constraint, or a guardrail around a learned human objective?
4. How will a critique-derived feature such as pass overlap be defined and validated?
5. How will the study distinguish discovering a fixed preference from the human genuinely revising a preference?
6. Can each comparison be traced through interpretation, intervention, and outcome?
7. What raw feedback and run artifacts are available for reconstructing the existing findings?

## Short meeting description

> The branch already has a working outer feedback loop around a differentiable trajectory optimizer. A human compares two toolpaths and explains the choice; an external coding agent maintains a textual preference interpretation and uses it to select or construct later experiments. The optimization itself directly adjusts tool movements through a smooth CSG simulator and evaluates the result with a sharp carve. What is not yet formalized is the coadaptive part: the system does not explicitly represent an evolving human state, distinguish changed preferences from noisy evidence, or validate newly inferred objectives such as surface finish and pass overlap.

# SSH Codex Handoff: Direct Free-Form Feedback Experiment

## Purpose

This document transfers the working context from a separate Codex task to a fresh Codex instance running on the GPU/SSH machine that contains the `diff-cam` repository. Treat it as a research handoff, not as proof that every architectural assumption below is correct. Verify claims against the checked-out code and record corrections.

## Immediate deliverables

Before the next lab meeting, approximately two days from the start of this task:

1. Understand the current `autoresearch` codebase and summarize the important attempts represented by earlier branches.
2. Set up and run an initial experiment in which a human's A/B choice plus free-form critique directly causes an LLM coding agent to alter the optimization objective and generate the next A/B pair.
3. Add factual text explanations for the two generated trajectories, emphasizing how they differ and what was changed in response to the critique.
4. Preserve a traceable log of every feedback-to-code-to-run transition.
5. Report what was implemented, what ran, whether the next trajectories changed meaningfully, failures/limitations, and the exact commands and artifacts needed to reproduce the result.

Do not claim that the method works merely because files changed or two videos differ. The first experiment asks whether the feedback produces a traceable and meaningful trajectory change.

## Research motivation

The reference free-form preference learning (FPL) work uses language to convey more information than an A/B label alone. A choice says which behavior is preferred; a critique can identify why—for example, excessive late air cutting, gouging, poor contour following, or jagged finish.

The intended experiment deliberately skips a conventional learned reward-model intermediary. Instead:

```text
Human chooses A/B/tie and writes a critique
        -> LLM coding agent reads that event
        -> agent changes the loss/objective implementation or its active terms
        -> agent records exactly what it inferred and changed
        -> two new, distinguishable trajectories are trained
        -> each option receives a factual explanation grounded in its actual
           configuration and measured outcomes
        -> human sees the next pair and responds again
```

The log is an audit trail, not the mechanism that translates feedback into the objective. The agent should act on the original feedback event and then write its decision record.

Longer-term framing is coadaptive value alignment: the displayed trajectories and their explanations may change what the human notices or values. For this initial experiment, simply preserve the exposure history so later work can study that effect. Do not attempt to solve the full latent-human-state problem this week.

## Codebase facts already established; verify them

The earlier review examined `origin/autoresearch` at commit `41b7355f05a6315bf2af47ee478501e519ca9232`.

### Optimizer

- `algorithms/train_csg.py` is GradMill's main program.
- It does not train a policy network. It directly optimizes approximately `T-1` three-dimensional tool-displacement vectors with Adam.
- A Taichi autodiff tape differentiates through the sequence of smooth CSG cuts into `sim.tool_delta.grad`.
- `simulator/csg_simulator.py` contains the differentiable and sharp cutting simulations and the loss components.
- Training uses smooth CSG operations; evaluation uses a sharp Boolean carve. This creates a soft/hard gap.
- `k` annealing and best-on-hard checkpoint selection address that gap.
- Existing terms include residual stock, gouge, holder collision, air/air-time/late-air, trajectory proximity, path length, jerk, step regularity, total time, direct tool-position gouge, and a simplified breakage-risk surrogate.
- Surface finish, scallop height, and pass-overlap quality do not appear to have an adequate direct objective.

### Existing preference loop

- `autoresearch/tasks/train_csg/web/compare.html` is the synchronized A/B/tie interface with a free-form “Why?” field.
- `scripts/serve_web_https.py` serves the UI and persists pairwise responses.
- `scripts/enqueue_pair.py` writes comparison pairs.
- `autoresearch/tasks/train_csg/sweep_pref_pair.sh` runs two configurations differing in a selected objective setting and enqueues them.
- `scripts/pref_lib.py` and `scripts/pref_digest.py` count A/B/tie answers per dimension and surface recent notes.
- `autoresearch/tasks/train_csg/autoresearch.md` instructs an external coding agent to interpret feedback and choose/edit later experiments.
- `autoresearch/tasks/train_csg/pref_objective_plan.md` is the prose “current preference understanding.”
- `autoresearch/tasks/train_csg/web/index.html` contains a separate star-rating and free-text system. `--use-feedback` can warm-start from a highly rated prior trajectory; do not confuse this with the pairwise critique loop.
- In `train_csg.py`, pairwise feedback is currently described as a “steering signal only, not in loss.” The direct bridge from an individual critique to a specific objective edit and then to the next pair is not implemented as a clearly traceable automated mechanism.

### Existing caveats

- The digest aggregates “prefers A” versus “prefers B” instead of aggregating the selected configuration/magnitude.
- A/B presentation order is not randomized.
- The interface reveals the manipulated dimension and magnitudes, which can anchor critique.
- Pair selection is heuristic rather than a formal uncertainty/information-gain strategy.
- The command-line enqueuer and server appear to use different process-local locks while writing the same JSON file, creating a possible lost-update race.
- Raw `pairwise.json`, run feedback, and most artifacts are gitignored, limiting provenance unless explicitly archived elsewhere.
- Hard Dice is used as the main advancement gate even when the human comments on qualities it does not measure.
- A repository `resources` symlink may point to another machine's absolute path; verify whether it is needed before changing it.

## Source material from the prior task

If these files are copied to the SSH machine, read them before proposing the experiment:

- `AUTORESEARCH_CODEBASE_README.md`: personal map of the `autoresearch` implementation.
- `MEETING_REFERENCE.md`: notes, framing, and questions from the literature/code review.
- `Relevant Papers/Freeform_Preference_Learning_for_Robotic_Manipulation_Torne.pdf`: primary FPL reference.
- `Relevant Papers/Coadaptive_Value_Alignment_Tsoi_2026.pdf`: coadaptive framing.
- `Relevant Papers/OVPR_New_Directions___Differentiable_Simulation_for_Manufacturing.pdf`: project direction.
- `Text-Based Critques.pdf`: draft text-critique material; ignore unrelated social-robot-navigation material.
- `Brief Outline Fall CS 370.docx`: semester outline.

If some documents are unavailable, continue from this handoff and the repository documentation. Clearly list which sources were unavailable.

## Required work sequence

### Phase 0: Protect the existing work

Before edits:

1. Print the repository root, current branch, exact commit, remotes, status, GPU inventory, disk space, and relevant runtime versions.
2. Fetch remote references without overwriting local work.
3. Do not reset, clean, discard, or overwrite uncommitted changes.
4. Create a dedicated experiment branch only after confirming the working tree is safe. Use a descriptive branch name and record its base commit.
5. Keep raw run artifacts out of Git if large, but store a small manifest with paths and checksums.

### Phase 1: Understand the branch history

Do not infer history only from branch names. Use commit graphs, merge bases, log summaries, diffs, documentation changes, and relevant code changes.

Review at least:

- `master`
- `sim-features`
- `sim-logic`
- `nt-cleanup`
- `cleanup`
- `aditya`
- `aditya-autoresearch`
- `autoresearch`
- the `ar-agd/*` branches, especially:
  - `jun28-decay-port`
  - `jun30-new-research`
  - `jul1-uniform-toolpath`
  - `jul3-hard-carve-gap`
  - `jul4-hard-carve-gap`
  - `jul4-toolholder-collision`
  - `jul5-anneal-gap`
  - `jul6-step-detail`
  - `jul6-traj-quality`
  - `jul7-balanced`
  - `jul8-multidepth`
  - `jul10-gouge`
  - `jul13-phys-plausible`

Produce `BRANCH_HISTORY_REVIEW.md` containing, for each meaningful line of work:

- Base/relationship to `autoresearch`
- Research question or failure mode
- Code/objective/init change attempted
- Evidence or metrics recorded
- Whether the result was retained, rejected, or remains uncertain
- Files and commits supporting the conclusion

Distinguish verified evidence from conclusions written in commit messages. Avoid rerunning old expensive experiments unless needed for the new baseline.

### Phase 2: Establish a runnable baseline

1. Read `README.md`, `SETUP_TACC.md`, `pyproject.toml`, `uv.lock`, `train_hpc.bash`, `train_local.bash`, and relevant task scripts.
2. Inspect the actual scheduler/GPU environment; do not assume commands intended for another cluster work here.
3. Resolve dependencies with the repository's intended environment manager where possible.
4. Run the smallest meaningful smoke test first.
5. Run one known baseline configuration and confirm that it produces metrics, a saved trajectory, and a viewable/renderable artifact.
6. Record the command, commit, environment, random seed, target shape, runtime, GPU, output directory, and headline metrics.

Do not launch a two-day sweep until the smoke test and one baseline succeed.

### Phase 3: Design the direct-feedback MVP

First trace the current request path from the UI POST through storage and into the agent/experiment process. Then implement the smallest reliable loop that demonstrates the desired causal chain.

The preferred event model is append-only or transactional. Each iteration needs a stable ID and should record:

- Pair shown and exact run IDs
- A/B display order
- What text/metrics were shown to the human
- Choice and raw critique
- Timestamp and, if feasible, first-view timestamp
- Agent's structured interpretation
- Exact objective terms or code altered
- Before/after values and source diff
- Rationale and uncertainty
- Hypothesis for option A
- Hypothesis for option B
- Training commands, seeds, and output paths
- Measured metrics and trajectory-difference checks
- Generated explanation for each option
- Next pair ID
- Git commit before and after, or a patch hash if commits are inappropriate

The raw critique must be the agent input. Write the audit event after deciding the change; do not make the digest/summary log the only input to the change.

### Phase 4: Define what A and B mean

Resolve this explicitly before training. Recommended initial design:

- Both options respond to the same critique.
- **A** applies a conservative interpretation or smaller objective change.
- **B** applies a stronger or alternative interpretation.
- Hold target shape, initialization family, seed, iteration count, hardware assumptions, and unrelated weights fixed wherever possible.

For the cleanest causal baseline, consider one additional control outside the displayed pair: rerun the prior objective with the same seed. This separates ordinary stochastic variation from feedback-induced change.

An alternative design is A = unchanged objective and B = feedback-adjusted objective. This is easier to interpret causally but does not satisfy “two newly adapted options” as directly. If choosing it, state the tradeoff.

Do not let A and B become arbitrary bundles of unrelated edits. The log must explain the intended distinction.

### Phase 5: Generate grounded explanations

For each trajectory, generate a short factual explanation from:

- The exact objective/configuration change
- The critique it responds to
- Actual measured metrics after training
- Any visible behavior that can be verified from trajectory-derived measurements

Separate predictions from observations. Example:

```text
Option A — conservative late-air penalty
Change: increased the late-air term from 0.001 to 0.003 while leaving the
geometry terms unchanged.
Intent: reduce unproductive motion near the end without strongly disturbing
the cutting path.
Observed: late-third air time fell from X to Y; hard Dice changed from P to Q.
```

Do not say that a trajectory is smoother, safer, or better unless a corresponding metric or inspected artifact supports it. Explanation text is itself an intervention on the human, so preserve its exact wording and display order in the event log. Use neutral language for this initial experiment rather than persuasive language.

### Phase 6: Enforce a meaningful-difference check

Before presenting the next pair, verify that A and B are distinguishable in a way related to the feedback. At minimum report:

- Difference in the targeted metric
- Difference in hard Dice and gouge/residual
- Difference in air time, total time, and breakage proxy where applicable
- A trajectory-space difference such as mean/max point displacement or path distance
- Whether the rendered paths are visually distinguishable

Set a declared threshold before seeing the result. If the pair is effectively identical, do not silently present it as a successful adaptation. Log it as a failed/weak intervention, strengthen or redesign the contrast once, and preserve both attempts.

### Phase 7: Run the initial experiment

Use one target shape and one feedback concept first. Prefer a concept already represented by stable metrics, such as late air cutting or direct gouge avoidance, before attempting an entirely new surface-finish objective.

Suggested first iteration:

1. Produce or select an initial pair.
2. Collect a real human choice and raw critique through the web UI.
3. Trigger the agent's direct interpretation path.
4. Produce two controlled objective variants.
5. Run the variants on the GPUs.
6. Generate grounded A/B explanations.
7. Check meaningful difference and enqueue the next pair.
8. Collect at least the next response if time permits, demonstrating that the loop can continue.

If live human input is unavailable while building, use a clearly labeled fixture only to test plumbing. Do not count fixture feedback as experimental evidence.

## Implementation boundaries

- Keep the first implementation narrow and reversible.
- Preserve metric integrity; do not modify hard Dice or other evaluation code to make a result look better.
- Do not allow unconstrained repeated LLM edits without versioning, validation, and a rollback point.
- Prefer configuration/weight changes for the first successful end-to-end test. Support source-level loss edits only through explicit patches that are syntax-checked and reviewed before GPU execution.
- Separate orchestration code from the core simulator where practical.
- Avoid requiring the human to manually copy feedback from the UI into the agent once the MVP is connected.
- If the installed Codex CLI must be invoked by a watcher/orchestrator, inspect the installed CLI's supported noninteractive interface rather than guessing flags. Record the exact invocation and permissions.
- Do not expose API keys, tokens, private paths, or scheduler credentials in logs or commits.

## Minimum success criteria

The initial experiment succeeds only if all of the following are demonstrated:

1. A human submits a choice and free-form critique.
2. The agent consumes that exact feedback event rather than only a manually rewritten summary.
3. A versioned objective/configuration change is attributable to the event.
4. Two controlled new trajectories are generated.
5. Their intended distinction is documented before or at launch.
6. Their actual distinction is measured after training.
7. The UI presents grounded explanatory text for both.
8. The complete chain can be reconstructed from logs and saved artifacts.

It is acceptable for the first objective change to fail to improve the trajectory. A traceable negative result is more valuable than an untraceable apparent success.

## Two-day report

Produce `INITIAL_DIRECT_FEEDBACK_EXPERIMENT.md` with:

- Executive result: what worked and what did not
- Environment and exact code revision
- Branch-history summary and pointers to `BRANCH_HISTORY_REVIEW.md`
- Baseline configuration and result
- Feedback event and agent interpretation
- Exact loss/config changes for A and B
- Source diff or parameter diff
- Commands and run IDs
- A/B generated explanations
- Metric and trajectory-difference table
- Links/paths to renders, videos, metrics, logs, and manifests
- Whether the pair was meaningfully different under the predeclared check
- Known confounds, especially stochastic variation
- Next experiment and unresolved questions

## Questions that require explicit answers during implementation

1. What process is the persistent LLM agent, and how is it triggered by a newly answered pair?
2. Does “directly alter the loss” include changing loss weights/configuration, or must it edit loss source code? Start with weights unless source editing is necessary to express the critique.
3. Should A/B be conservative versus strong adaptations, competing interpretations, or unchanged versus adapted?
4. Which target shape and initial critique will provide the fastest interpretable demonstration?
5. What numeric/trajectory threshold counts as a noticeable change before the pair is shown?
6. Which GPU/scheduler commands and runtime limits apply on this host?

If these cannot be determined from the code or environment, ask the user concise questions before launching expensive jobs. Continue all safe inspection and setup work in parallel with waiting for answers.

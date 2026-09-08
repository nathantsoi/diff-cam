# Phase 4 Controlled Variant Design

## Decision

Both displayed children are feedback-adapted versions of the same selected
parent:

- **A:** conservative change to one or two existing loss weights.
- **B:** stronger change to the same weights in the same direction.

The parent is the human-selected run. A tie deterministically selects A and
records that rule. Shape, seed, iteration count, trajectory length, voxel size,
initialization, geometry, optimizer settings, checkpoint criteria, and all
unrelated loss weights are copied from the parent's saved reproduction command.

This is objective-level meta-optimization: the LLM changes the loss used by
direct trajectory optimization. It does not update an RL policy.

## Implementation

`scripts/run_direct_feedback_variants.py` consumes an
`objective_decided` iteration from Phase 3. It:

1. Loads the parent `reproduce_command.sh` rather than reconstructing hidden
   defaults.
2. Tokenizes the command without a shell.
3. Removes W&B tracking and the legacy star-feedback warm start.
4. Applies only the validator-approved loss values.
5. Gives A and B separate run subdirectories.
6. Parses both commands and refuses the plan unless their only differences are
   the targeted loss values and output path.
7. Refuses code drift between the Qwen decision and launch planning. Both
   decision and planning also require a clean, fully versioned worktree.
8. Writes `launch_plan.json`, its SHA-256, exact commands, hypotheses, fixed
   controls, and the predeclared meaningful-difference thresholds.

The HTTPS server invokes this planner automatically after a successful Qwen
decision. It does not pass `--execute`, so answering a browser prompt cannot
silently launch an expensive GPU job during Phases 3–4.

## Predeclared meaningful-difference thresholds

- At least 20% relative change in the metric targeted by the critique.
- At least 0.5 mm (one voxel in the approved scenario) mean pointwise
  trajectory displacement between A and B.
- Neither adaptation may reduce hard Dice by more than 0.03 relative to the
  selected parent for a successful quality-preserving claim.
- Gouge may not increase by more than 10% relative to the selected parent for a
  successful no-material-regression claim.

Phase 6 will report all values even when these gates fail. If A and B are too
similar, the attempt is logged as weak and may be strengthened once; it is not
presented as successful evidence.

## Execution interface

Planning only (normally automatic after the Phase 3 worker):

```bash
.venv/bin/python scripts/run_direct_feedback_variants.py \
  --iteration-id <df_pair_timestamp>
```

Explicit sequential execution on one inspected idle GPU:

```bash
.venv/bin/python scripts/run_direct_feedback_variants.py \
  --iteration-id <df_pair_timestamp> --execute --gpu <physical-index>
```

Sequential use of the same GPU avoids asymmetric contention. The runner logs
each command's combined output separately, captures run IDs and complete
metrics, and appends start/completion records to the event stream.

## Verification

The focused test suite now has 11 passing tests. Phase 4 tests verify that:

- the parent command is cloned;
- A/B seed and scenario remain identical;
- only the approved loss term and output path differ;
- W&B and legacy warm-start flags are removed;
- code drift is rejected; and
- execution requires an explicit GPU.

No synthetic feedback is treated as human evidence, and no Phase 4 GPU run has
been launched yet.

## Remaining prerequisite for the initial experiment

The approved downstream scenario is the Phase 2 sphere setup (seed 1, 5,000
iterations, 128 steps, 0.5 mm voxels). A real initial A/B observation still has
to identify the selected parent and provide the raw critique. Once that event
exists, the direct agent and planner can produce the concrete experiment ID and
commands for review before the approximately two 40-minute sequential runs.

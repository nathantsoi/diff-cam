# autoresearch

You are an AI Research assistant. Your objective is to find the best method of training the analytical gradient descent approach by performing experiments — including making it work well **across different stock and target shapes/sizes**, not just one fixed scenario. **Crucially, your optimization and initialization methods must generalize to arbitrary, unseen, and unique combinations of shapes: you MUST NOT allow the optimizer, initializations, or loss functions to inspect, branch on, or leverage the task type or shape name (e.g., whether `--target-shape` is `sphere`, `cylinder`, `box`, or `pyramid`). All algorithms must operate blindly to task metadata and rely solely on the geometric representation in the simulator.**

## The machining scenario (stock & target)

The normalized geometry cube `[0,1]^3` is the **stock box** (the raw block, the only thing voxelized); the machine work volume (Haas Mini Mill, 16x12x10 in) is separate metadata. The **target** is the part to be carved inside the stock. Both are configurable from the CLI:

- `--stock-size-in X Y Z` — stock box in inches (default `1 1 1`; can be non-cubic, e.g. `2 1 1`).
- `--voxel-size-mm F` — physical voxel edge in mm, the sub-mm precision knob (default `0.5`). RAM scales with the stock volume, so finer voxels / larger stock cost more VRAM.
- `--target-shape S` — `sphere` | `cylinder` | `box` | `pyramid`.
- `--target-radius-mm F` — sphere/cylinder radius, or box/pyramid half-size (mm).
- `--target-height-mm F` — cylinder/pyramid height (mm); ignored for sphere/box.
- `--stock-origin-in X Y Z` — work origin (G54) = stock top-centre in machine inches (export/validation only; does not affect dice).

Size the target to fit **inside** the stock (1 in = 25.4 mm). The dice score is the carved-stock-vs-target overlap, so the target choice defines the task — when you change the scenario, runs are only comparable to others with the **same** stock/target config. Use scenario changes to test generality and find hard cases, and validate any *method* improvement on the default scenario before claiming it as a win.

## Setup

To create a new experiment:

1. **Choose run tag**: choose a new tag for your experiment that includes the date and a unique word description (e.g. `jun27-random-init`). The branch `ar-agd/<tag>` must not already exist, this is a fresh run.
2. **Create the branch**: `git checkout -b ar-agd/<tag>` from the `autoresearch` branch.
3. **Clear the previous run's results**: the result artifacts from the last run still sit in `autoresearch/tasks/train_csg/` and will contaminate the new branch if not cleared — a new branch must start from a clean slate, with results recorded only for *this* run. Once you are on the new branch, reset/truncate them:
   - `results.tsv` → truncate to just the header row (`commit\tdice\tmemory_gb\tstatus\tdescription\tcommand`). Do NOT carry over any prior experiment rows. (Leave it untracked — do not commit it; see "Logging results".)
   - `idea.md` → overwrite with a fresh file (step 6 below) noting the new branch/tag, starting point, and plan. The old chronological working log does not apply to this run.
   - `findings.md` → `rm` it. It is the *consolidated findings record* from the previous run and is regenerated only when this run concludes; leaving it around would report stale numbers as if they were this run's. (The prior run's findings are preserved on its own branch in git history and in memory.)
   - `rm -f results_plot.png run.log` (both untracked) so the plot/log don't mix runs.
   These are committed-on-the-new-branch changes (or untracked deletions); do not push them back to the prior branch. The prior run's results remain preserved on its own branch in git history.
4. **Create the run-output folder**: `mkdir -p runs/<tag>` (e.g. `runs/jun27-random-init`). Every experiment on this branch must land **inside** `runs/<tag>/` — the results dashboard groups runs by these direct child folders of `runs/` (see `discover_batches` in `scripts/build_results_web.py`), so a branch's experiments only appear together as one batch if they share the folder. The trainer writes to `runs/<run_name>` at the top level by default, so after each run you must move it into `runs/<tag>/` (done in the experiment loop below).
5. **Read the in-scope files**: Start with the train_csg.py file and README.md
6. **Create a new autoresearch/tasks/train_csg/idea.md file** to record your ideas and take notes. Summarize your most interesting findings in this file as they arise.

## Experimentation

Run each experiment on a single GPU, you can run multiple experiments at once, but check GPU load first and distribute accordingly by setting CUDA_VISIBLE_DEVICES. The training script runs for a **fixed time budget of no more than 15 minutes** (wall clock training time, excluding startup/compilation). Launch training as: `uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10`

`--voxel-size-mm` is the precision knob (a 1 in cube at `0.5` → 51³ grid, ~0.14 GB). Two levers dominate everything else:

- **`--learning-rate 1e-3` is the single biggest lever** (the code default is still the old `5e-3` — you MUST pass `--learning-rate 1e-3` explicitly). The old `5e-3` **overshoots** past the good carving basin (dice peaks mid-training then degrades); `1e-3` lets the optimizer **settle** into the basin, producing a higher *and sustained* peak (final-iter ≈ best). Monotonic on sphere: `5e-3→0.717, 2e-3→0.754, 1e-3→0.849 (peak), 5e-4→0.804 (underfits)`. Sharp unimodal peak at `1e-3`; **lr is exhausted** — do not re-sweep. Universal across shapes (sphere 0.67→0.84, box 0.84→0.92, pyramid 0.86→0.89, cylinder 0.74→0.92). At `1e-3` the "transient peak then degrade" that motivated best-checkpoint saving is largely gone, but keep best-checkpoint saving on (cheap insurance).
- **`--dt 0.45`** is the foundational lever (do NOT use the old `≈ 0.12` advice): at low dt the swept-cylinder tool is speed-limited and cannot descend/traverse the part exterior, capping dice at ~0.56. `dt=0.45` advances ≈ 1 voxel/step so the tool covers the part. Sweet spot `dt ∈ [0.42, 0.5]` (but see dead levers: `dt0.5 + m160` is a single-seed fluke, not a real win).

`--grad-clip 0.5` + `--eval-freq 10` + best-checkpoint saving capture the dice peak. (The old "`--grad-clip 0.4` for sphere" advice is obsolete — at lr=1e-3, gc=0.4 drops sphere to 0.786; use gc=0.5 everywhere.) Per-shape refinements `--w-len 0.03` (trailing-drift fix), `--w-step 0.001` (uniform-feed), `--init-mode raster_fine`, and cyl `--max-steps 256` are documented in [Proven operating point & dead levers](#proven-operating-point--dead-levers).

**What you CAN do:**
- Modify `train_csg.py`, parameters when you call this script, and related components. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, max steps, etc. Importantly, if you need to implement new loss components, this will need to be done in the differentiable simulator, which is allowed.
- Fix bugs and improve model training overall.
- **Vary the machining scenario** via the CLI flags above: change the stock shape/size (`--stock-size-in`, including non-cubic blocks), the voxel precision (`--voxel-size-mm`), and the target shape/size (`--target-shape`, `--target-radius-mm`, `--target-height-mm`). Explore whether the method holds up across spheres, cylinders, boxes, pyramids, and different stock sizes — and tune the method to do well across them.

**What you CANNOT do:**
- **Leverage task metadata or shape names in the optimizer or initialization**: Do NOT allow the optimization process, trajectory generator, loss functions, or initialization algorithms to inspect, branch on, or leverage the task type/name (e.g., whether `--target-shape` is `sphere`, `cylinder`, `box`, `pyramid`, or any shape string). The optimization approach must operate **blindly** to the target shape name and rely *solely* on the general physical and geometric representation in the simulator (e.g., voxel occupancy grid $\Phi$, target SDF $\phi_{\mathrm{tgt}}$, bounding boxes, spatial collision queries, or gradient fields). This is critical to ensure that the method generalizes robustly to arbitrary, unseen, and unique combinations of shapes without hand-coded per-shape heuristics or branches.
- Install new packages or add dependencies.
- Modify components of the simulator that change *how* the evaluation is computed (the dice/ASD/HD95 metric code, the carve used for scoring). Changing the stock/target **via the CLI flags** is allowed and encouraged — that selects the task; editing the metric/eval code to inflate the score is not.
- Modify the evaluation harness.

**The goal is simple: get the highest dice score** — on the default scenario, and ideally robustly across the stock/target scenarios you explore (compare dice only within the same scenario). Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful dice gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 dice improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 dice improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline — run the pipeline script with the default scenario (the baseline command in [Experimentation](#experimentation): default 1 in cube stock, sphere target, `--voxel-size-mm 0.5`, and **`--learning-rate 1e-3`**) and no method changes. All later scenario or method variations are compared against this. **Note**: most of the proven operating point is baked into the code defaults (`--dt 0.45 --grad-clip 0.5 --eval-freq 10 --iters 5000`), **but `--learning-rate` is NOT — its code default is still the old `5e-3`**, so you must pass `--learning-rate 1e-3` explicitly or the run overshoots and scores ~0.67 (sphere) instead of ~0.85. With `--learning-rate 1e-3`, a fresh baseline scores **~0.85 (sphere) / ~0.92 (box) / ~0.89 (pyramid) / ~0.92 (cylinder)** reliably — compare new ideas against this strong basin. See [Proven operating point & dead levers](#proven-operating-point--dead-levers) for what is already established and what has been ruled out.

## Output format

Once the script finishes it prints a summary like this:

```
---
dice:              0.852300
asd:               1.234500
hd95:              3.456700
loss:              0.012345
residual:          0.010000
gouge:             0.002345
holder_overlap:    0.000000
training_seconds:  12.340000
peak_vram_mb:      1024.500000
num_steps:         128
---
```

Note that the script outputs clean scrolling lines by default (without `tqdm` carriage-return overwriting) so that LLM harnesses like Claude Code can easily ingest logs. You can extract the key metric from the log file:

```
grep "^dice:" run.log
```

Alternatively, the full summary metrics are written to JSON format inside the run directory at `runs/<run_name>/metrics.json` and a static copy is saved to `runs/latest_metrics.json`.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	dice	memory_gb	status	description	command
```

1. git commit hash (short, 7 chars)
2. dice achieved (e.g. 0.852300) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried
6. the exact run command used (the full `uv run python scripts/run_pipeline.py ...` invocation, WITHOUT the `> run.log 2>&1` redirect). This captures the scenario (stock/target/voxel flags) and hyperparameters so every row is reproducible and so the plot can group by config. It must contain no literal tab characters.

Example:

```
commit	dice	memory_gb	status	description	command
a1b2c3d	0.849800	0.1	keep	baseline sphere (lr1e-3)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10
b2c3d4e	0.854700	0.1	keep	sphere + w_len 0.03	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --w-len 0.03
c3d4e5f	0.920600	1.1	keep	cylinder (lr1e-3, w_len, T256)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 256 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape cylinder --target-radius-mm 11.43 --target-height-mm 22.86 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --w-len 0.03
d4e5f6g	0.000000	0.0	crash	0.1mm voxels on 2in cube (OOM)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 2 2 2 --voxel-size-mm 0.1 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `ar/mar5` or `ar/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `algorithms/train_csg.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment, redirecting everything to `run.log` (do NOT use tee or let output flood your context). Record the exact command you ran (see "Logging results"). Example:
   `uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 > run.log 2>&1`
5. **Move the run into the branch folder** so the dashboard groups it with this branch's batch: extract the run dir the trainer reported (`grep "writing outputs to" run.log` → `runs/<run_name>`) and `mv runs/<run_name> runs/<tag>/`. This must happen before the next experiment so runs don't pile up at the top level of `runs/` (where `discover_batches` can't see them). The run-dir matching in `build_results_web.py` keys on (shape, iters, seed) + dice, not path, so moving is safe.
6. Read out the results: `grep "^dice:\|^peak_vram_mb:" run.log` or read `runs/latest_metrics.json`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv, including the exact run command in the `command` column (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If dice improved (higher), you "advance" the branch, keeping the git commit
10. If dice is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~15 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~15 minutes then you can run approx 12/hour, for a total of about 50 experiments over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Concluding the research loop (make the plot)

The loop is autonomous and never self-terminates — but whenever it *is* wound down (the human interrupts you and asks you to wrap up, or you are otherwise concluding the entire research effort), you MUST produce a summary plot **before** you give your final report. Do not conclude without it.

Generate the plot from `results.tsv` (the source of truth — it has the per-experiment dice and the run command). Use `matplotlib` (already available; it is what the trainer uses for `metrics.png` — do NOT install anything). Write a small script to `autoresearch/tasks/train_csg/plot_results.py` and save the figure to `autoresearch/tasks/train_csg/results_plot.png`. The figure should make the research legible at a glance:

- **Progress over experiments**: dice on the y-axis vs. experiment order on the x-axis, with kept vs. discarded vs. crashed points distinguished, and a "running best" line so the improvement trajectory is obvious.
- **Generality across scenarios**: since you vary the stock/target, also show the **best dice per machining scenario** (parse the stock/target flags out of the `command` column — e.g. group by `--target-shape` and `--stock-size-in`). A grouped bar chart is fine. This is the payoff of varying the scenario: it shows where the method is strong and where it struggles.

Keep the script simple and robust (skip `crash` rows / `0.000000` dice where appropriate, handle a commit appearing more than once by taking its best). Then summarize the plot's takeaways in `idea.md` and in your final message. Finally, write `autoresearch/tasks/train_csg/findings.md` — the consolidated, deduplicated record of every finding from the run (task, best method, the real levers in order of impact, dead levers, methodological lessons, artifacts), drawing on `idea.md` (chronological) and `results.tsv` (numbers). This is the file the next run's `autoresearch.md` "Proven operating point" section will be built from, so be precise about configs and effect sizes. After the plot and `findings.md` exist, you may conclude.

## Proven operating point & dead levers

The `ar-agd/jul1-uniform-toolpath` effort (~127 experiments; see `findings.md` and the prior `idea.md`/`results.tsv` on that branch) established the operating point below, superseding the older `jun28-decay-port` table (whose `lr=5e-3` advice is now known to be suboptimal). Most of it is baked into the code defaults (`scripts/run_pipeline.py` + `algorithms/train_csg.py`), **except `--learning-rate`** (code default is still `5e-3` — pass `1e-3` explicitly). Use this as the reference point; do not waste runs re-discovering what is already known.

### Operating point (the best method, all shapes)

```
--dt 0.45 --learning-rate 1e-3 --init-mode raster_fine --w-len 0.03 \
--max-steps 256 --grad-clip 0.5 --eval-freq 10 --iters 5000
```

| lever | value | code default | why |
|-------|-------|--------------|-----|
| `--learning-rate` | `1e-3` | `5e-3` ❗ (must pass flag) | **The single biggest lever.** Old `5e-3` OVERSHOOTS past the good carving basin (dice peaks then degrades); `1e-3` lets the optimizer SETTLE — higher *and sustained* peak. Monotonic on sphere: `5e-3→0.717, 2e-3→0.754, 1e-3→0.849 (peak), 5e-4→0.804`. Sharp unimodal peak; lr EXHAUSTED. Universal across shapes. |
| `--dt` | `0.45` | `0.45` ✓ | Foundational lever. dt=0.12 caps dice ~0.56 (tool too slow to cover); 0.45 ≈ 1 voxel/step. Sweet spot `dt ∈ [0.42, 0.5]`. |
| `--w-len` | `0.03` | `0.0` ❗ (pass for cyl) | Path-length / minimal-motion penalty (mean sq `|Δ_t|²`). **The fix for "tool moves away from the part"** — agnostic to *where* the tool is (unlike the dead contour-hug losses), only discourages motion; trailing steps with no residual shrink to zero so the tool STOPS. Cylinder: trailing z-climb 1.704→0.010, dice 0.934→0.945, air 0.286→0.199. Safe for sphere (0.847→0.855). |
| `--w-step` | `0.001` | `0.0` ❗ (pass for sphere) | Constant-feed regularizer (squared step-LENGTH change). Encourages the uniform-feed CNC pattern without opposing carving. Saturates fast (0.001 enough). Sphere +0.004 mean — **marginal** (single-seed 0.858 was a lucky high-variance seed). Orthogonal to `w_len` but the two overlap (both discourage motion), so they do not stack. |
| `--init-mode` | `raster_fine` | `random` ❗ (optional) | Clipping-aware fine boustrophedon (per-step ≤ feed cap). +0.063 sphere mean vs random at lr=5e-3. At lr=1e-3 the LR win largely subsumes it (random + lr1e-3 ≈ rf + lr1e-3 on sphere) — kept as the uniform-feed init, but unnecessary at the correct LR. |
| `--max-steps` | `128` (256 for cyl) | `128` ✓ | m=128 optimal at dt≤0.45; cyl soft-dice peaks T=256 (0.9457, marginal over T=128). m≥192 NaNs (SDF overflow); m=144 slightly worse; T=320 unstable. |
| `--grad-clip` | `0.5` | `0.5` ✓ | Stabilizes the peak so best-checkpoint saving captures a higher one. **Use 0.5 everywhere** — the old "0.4 for sphere" advice is obsolete (at lr=1e-3, gc=0.4 drops sphere to 0.786). |
| `--eval-freq` | `10` | `10` ✓ | Fine cadence samples the peak. |
| `--iters` | `5000` | `5000` ✓ | Sweet spot. iters>5000 is marginal (cyl 10k→0.9477, +0.002 at 2× compute) — coverage-capped, not iters-limited. |
| best-checkpoint saving | on | on ✓ | At lr=1e-3 the peak is largely sustained, but keep it on (cheap insurance). No re-eval — GPU atomic-add nondeterminism gives ±0.01–0.05 run-to-run variance. |

**Seed variance strategy**: high run-to-run variance (±0.04–0.05) from init stochasticity + GPU atomic-add nondeterminism. **Need ≥3 (ideally ≥5) paired same-GPU seeds** to distinguish a real lever from seed-reshuffling — single-seed apparent wins overstate effect size ~2–3× (this bit the prior run on `w_step`, `w_gouge`, `dt0.5+m160`). **Dice is only comparable on the SAME GPU** (atomic-add nondeterminism); cross-GPU comparisons are confounded.

### Dead levers (confirmed no help — do NOT re-explore)

- **`w_air`, `w_prox`, `w_traj_prox`** (loss-based air / "keep tool near surface") — FUNDAMENTALLY trade off dice (0.847→~0.55) across all weights AND warmup. The default sphere nearly fills the 1in stock, so carving the corners inherently requires sweeping through empty corner region far from the surface → ~30% air-cut fraction is the *price* of 0.847 dice, not a tunable inefficiency. The air fix is `w_len` (stops trailing wandering) + `raster_fine` init (pre-covers the part), NOT a loss term. Code remains gated at default 0.
- **`w_gouge`** sweep (4→8) — seed-reshuffling, not a real mean win over 5 paired seeds (mean identical 0.6845; reshuffles dice across seeds). The 3-seed "+0.011 mean" was subset variance.
- **`w_jerk`** — ~neutral at every weight tested.
- **`lr_decay_frac`** — dead (all settings tied the baseline); best-checkpoint saving subsumes it.
- **`dt0.5 + m160`** — single-seed fluke (s0=0.853); mean 0.834 < dt0.45 m128 mean 0.849, higher air, higher variance.
- **`raster_fine_wide`** (full 0.05–0.95 envelope) — slightly worse mean, loses the high ceiling; under-coverage hypothesis refuted.
- **`k ≤ 2`** (sharp soft union) — saturates, gradients vanish, trajectory degenerates (soft dice 0.0). k=10 is correct.
- **`iters > 5000`** — marginal (cyl 10k→0.9477, +0.002 at 2× compute); cyl is coverage-capped.
- **finer `voxel_size_mm`** — 0.4 / 0.35 both WORSE than 0.5; the speed limit binds harder relative to voxel size.
- **coarse structured inits** (`raster`/`spiral`/`shell`/`zlayer`) — fail the speed clip (only `raster_fine` survives, because its per-step is ≤ feed cap).
- **`init-scale`** 0.02 / 0.1 — both hurt. **`lr`** 3e-3 (neutral), 7e-3 (diverges), 5e-4 (underfits) — lr EXHAUSTED, peak at 1e-3. **`max-steps`** 144 (slightly worse), ≥192 (NaN).

### Per-scenario ceilings (default stock 1.0 in, voxel 0.5 mm, lr=1e-3)

| scenario | best soft dice | mean (multi-seed) | config | vs lr=5e-3 baseline |
|----------|----------------|--------------------|--------|---------------------|
| cylinder | **0.9499** | 0.941 | dt0.45 lr1e-3 w_len0.03 T256 | 0.74 → 0.94 (+0.18) |
| box | **0.9172** | 0.915 | dt0.45 lr1e-3 rf | 0.84 → 0.92 (+0.08) |
| sphere | **0.8498** | 0.847 | dt0.45 lr1e-3 (rf or random) w_step0.001 | 0.67 → 0.85 (+0.18) |
| pyramid | **0.9001** | 0.893 | dt0.45 lr1e-3 rf | 0.86 → 0.89 (+0.03) |

These are **soft**-dice ceilings (the tracked metric). **Critical caveat — the soft/hard carve gap (~0.21):** the tracked soft dice (~0.94 cyl) is a BIASED proxy; the true deployable HARD-carve dice is ~0.718 and is k-invariant, T-invariant (coverage-capped). The soft union over-erodes (adds ~log(2)/k per step), so a trajectory optimized for soft does NOT transfer to hard. Staged training (train → truncate at t* → save state → train a 2nd trajectory to finish) works end-to-end but gave only +0.0016 hard dice because stage-2's soft optimization doesn't transfer. **To raise DEPLOYABLE dice, improve the trajectory's hard-carve coverage (more steps / finer feed / better path), not the loss smoothness or soft dice.** This is the highest-value open frontier.

Pure seeding and the levers above have hit diminishing returns — the productive frontier is a *new method lever*: either close the soft/hard gap (a less-biased soft objective, or a trajectory that covers more hard material), a parametric low-air toolpath (inherently uniform CNC pattern), or lifting a structural cap on a specific shape. Not more seeds, not more lr/iters/w_len sweeps.

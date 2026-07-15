# autoresearch

You are an AI Research assistant. Your objective is to find the best method of training the analytical gradient descent approach by performing experiments — including making it work well **across different stock and target shapes/sizes**, not just one fixed scenario. You are free to modify **any editable element** of the method and training pipeline: the optimizer, initializations, training loss, model architecture, simulator differentiable functions, the metric/evaluation code, and the evaluation harness. Nothing in the codebase is off-limits to editing except adding new dependencies (see "What you CANNOT do"). Generalization across unseen shapes is still the *research goal* to strive for, but you are no longer forbidden from inspecting the target shape name or task metadata in the optimizer/init/loss — use your judgment about whether per-shape branching helps the method generalize or merely overfits to the shapes you test.

**A primary, ongoing focus of this run is eliciting human feedback via pairwise A/B comparisons in the web UI** (`web/compare.html`). The blunt `hard_dice` deployability gate cannot see trajectory qualities the human actually cares about — end-of-path air cutting, surface following, tool-break safety, finish — so you learn the real objective by presenting pairs of trajectories that vary one knob at two magnitudes and asking the human to pick the better one with a short reason. The comparison queue is **not** a side task you top up occasionally; **generating fresh comparison pairs is the main thing the loop does**, and the human's answers steer both your next objective formulation and (where it helps) loss terms encoded directly into training. Crucially, **the human may be away for hours** — you do NOT pause or stop between runs to wait for answers. **Keep generating comparisons continuously while waiting for the user**: whenever GPUs are free, produce the next A/B pair across a diverse rotation of dimensions / shapes / seeds / magnitudes, enqueue it, and immediately move on to the next. The queue should stay stocked so that whenever the human opens the UI there is always something fresh to judge; never block the loop on pending answers. (See step 0 of the experiment loop and `pref_objective_plan.md` for the mechanics.)

## The machining scenario (stock & target)

The normalized geometry cube `[0,1]^3` is the **stock box** (the raw block, the only thing voxelized); the machine work volume (Haas Mini Mill, 16x12x10 in) is separate metadata. The **target** is the part to be carved inside the stock. Both are configurable from the CLI:

- `--stock-size-in X Y Z` — stock box in inches (default `1 1 1`; can be non-cubic, e.g. `2 1 1`).
- `--voxel-size-mm F` — physical voxel edge in mm, the sub-mm precision knob (default `0.5`). RAM scales with the stock volume, so finer voxels / larger stock cost more VRAM.
- `--target-shape S` — `sphere` | `cylinder` | `box` | `pyramid` | `sphere_hole` | `sphere_bowl`. The first four are single CSG primitives; `sphere_hole` is a 0.9 in sphere with a 0.75 in through-hole cylinder subtracted, and `sphere_bowl` is a 0.9 in sphere with a lower hemisphere subtracted. The optimizer is shape-agnostic and trains on any of these from the baked target SDF alone (authoritative list: `simulator/csg_simulator.py::_init_target_fields`).
- `--target-radius-mm F` — sphere/cylinder radius, or box/pyramid half-size (mm).
- `--target-height-mm F` — cylinder/pyramid height (mm); ignored for sphere/box.
- `--stock-origin-in X Y Z` — work origin (G54) = stock top-centre in machine inches (export/validation only; does not affect dice).

Size the target to fit **inside** the stock (1 in = 25.4 mm). The dice score is the carved-stock-vs-target overlap, so the target choice defines the task — when you change the scenario, runs are only comparable to others with the **same** stock/target config. Use scenario changes to test generality and find hard cases, and validate any *method* improvement on the default scenario before claiming it as a win.

## Setup

To create a new experiment:

1. **Choose run tag**: choose a new tag for your experiment that includes the date and a unique word description (e.g. `jul8-<idea>`). The branch `arq-agd/<tag>` must not already exist, this is a fresh run.
2. **Create the branch**: `git checkout -b arq-agd/<tag>` from the `autoresearch` branch.
3. **Clear the previous run's results**: the result artifacts from the last run still sit in `autoresearch/tasks/train_csg/` and will contaminate the new branch if not cleared — a new branch must start from a clean slate, with results recorded only for *this* run. Once you are on the new branch, reset/truncate them:
   - `results.tsv` → truncate to just the header row (`commit\tdice\tmemory_gb\tstatus\tdescription\tcommand`). Do NOT carry over any prior experiment rows. (Leave it untracked — do not commit it; see "Logging results".)
   - `idea.md` → overwrite with a fresh file (step 6 below) noting the new branch/tag, starting point, and plan. The old chronological working log does not apply to this run.
   - `findings.md` → `rm` it if present. It is the *consolidated findings record* from a previous run and is regenerated only when this run concludes; leaving it around would report stale numbers as if they were this run's. (A prior run's findings are preserved on its own branch in git history.)
   - `rm -f results_plot.png run.log` (both untracked) so the plot/log don't mix runs.
   These are committed-on-the-new-branch changes (or untracked deletions); do not push them back to a prior branch. Prior run results remain preserved on their own branches in git history.
4. **Clear the project auto-memory**: the project's persistent auto-memory at `~/.claude/projects/-home-ntsoi-papers-icra26-diffcam-diff-cam/memory/` (a `MEMORY.md` index plus per-fact `.md` files) is loaded into context at the start of every Claude Code session in this repo — including this autoresearch loop — and records conclusions from prior runs (best configs, ruled-out levers, dice numbers, bug bisects). To avoid biasing a fresh run, archive it out of the loaded directory before you start: move the existing `*.md` files to a sibling archive folder (e.g. `memory_archive_<date>/`) and leave a fresh, empty `MEMORY.md` in `memory/`. Recoverable from the archive if needed.
5. **Create the run-output folder**: `mkdir -p runs/<tag>`. Every experiment on this branch must land **inside** `runs/<tag>/` — the results dashboard groups runs by these direct child folders of `runs/` (see `discover_batches` in `scripts/build_results_web.py`), so a branch's experiments only appear together as one batch if they share the folder. The trainer writes to `runs/<run_name>` at the top level by default, so after each run you must move it into `runs/<tag>/` (done in the experiment loop below).
6. **Read the in-scope files**: Start with `algorithms/train_csg.py` and `README.md`.
7. **Create a new `autoresearch/tasks/train_csg/idea.md`** to record your ideas and take notes. Summarize your most interesting findings in this file as they arise.

## Experimentation

Run each experiment on a single GPU. You can run multiple experiments at once, but check GPU load first and distribute accordingly by setting `CUDA_VISIBLE_DEVICES`. Allow each training run to run until convergence. Launch training as:

```
uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 \
  --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere \
  --target-radius-mm 11.43 --post haas --eval-freq 10
```

The command above is a **template** — the stock/target flags define the scenario, and every other flag (`--dt`, `--learning-rate`, `--grad-clip`, `--max-steps`, `--init-mode`, `--init-scale`, the `--w-*` loss weights, the `--best-w-*` checkpoint weights, `--k-init`/`--k-final`, `--voxel-size-mm`, etc.) is a **knob you are free and expected to tune**. Treat all hyperparameter defaults as starting points to explore, not as proven values. Discover what works by experimenting.

**What you CAN do:**
- Modify `algorithms/train_csg.py`, pass different parameters when you call the script, and modify related components. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, max steps, init strategy, loss components, etc. **The training loss is fully editable** — you may add/remove/reweight loss terms, encode human pairwise preferences into the loss, and even branch the loss/init/optimizer on the target shape name or task metadata if you judge it helps.
- Fix bugs and improve model training overall.
- Add optimization objectives and differentiable functions to the simulator.
- **Edit the metric and evaluation code too**: the dice/ASD/HD95 computation, the carve used for scoring, and the evaluation harness are all editable. (See the integrity warning below.)
- Vary the machining scenario via the CLI flags above: change the stock shape/size (`--stock-size-in`, including non-cubic blocks), the voxel precision (`--voxel-size-mm`), and the target shape/size (`--target-shape`, `--target-radius-mm`, `--target-height-mm`). Explore whether your method holds up across spheres, cylinders, boxes, pyramids, and the combined CSG shapes (`sphere_hole`, `sphere_bowl`), and different stock sizes — and tune the method to do well across them.
- Changing the stock/target via the CLI flags or via new/better implemenation, is allowed and encouraged.

**What you CANNOT do:**
- Install new packages or add dependencies. (This is an environment/dependency constraint, not an editability one — reuse what `uv` already resolves.)

**Integrity warning (you ARE allowed to edit the metric/harness, so police yourself):** because the eval metric/harness is now editable, the loop could "win" by rewriting `hard_dice` to score higher instead of by improving the method — which makes every subsequent comparison meaningless. When you edit metric/eval/harness code, do so to make the measurement *more honest* (fix a bug, add a deployability-relevant term, correct a computation), never to inflate the score. If a metric change alters what `hard_dice` means, re-baseline against the prior meaning before claiming an advance, and record the metric change explicitly in the TSV description so the human can see the scoreboard moved. Treat a method improvement that only shows up after a metric edit as unproven.

**The goal is simple: get the highest `hard_dice` score without breaking the tool in the shortest trajectory (and trajectory execution time) possible.**  Note that other measure and objectives including `air_time`,  `total_time`, and `break_prob_any` will help guide you to the objective.

**VRAM** is a soft constraint. Use as much or little as you need up to the limit of the hardware you have available. 

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run on a fresh branch should always be to establish the baseline — run the pipeline with the default scenario (the baseline command in [Experimentation](#experimentation): default 1 in cube stock, sphere target, `--voxel-size-mm 0.5`) and no method changes. All later scenario or method variations are compared against this. Read the in-scope files first to understand the defaults before running.

## Output format

Once the script finishes it prints a summary like this:

```
---
hard_dice:         0.718300
dice:              0.852300
dice_baseline:     0.419800
dice_improvement:  0.513200
asd:               1.234500
hd95:              3.456700
loss:              0.012345
residual:          0.010000
gouge:             0.002345
holder_overlap:    0.000000
training_seconds:  12.340000
peak_vram_mb:      1024.500000
num_steps:         128
air_time:          0.000000
total_time:        2.980263
break_prob_any:    0.000000
break_prob_max:    0.000000
fcut_max:          0.000000
broken:            0.000000
engage_max:        0.000000
engage_mean:       0.000000
best_score:        0.560100
---
```

**`hard_dice` is the FINAL, deployable metric and the number you optimize/advance on.** It is the sharp boolean carve (what the part actually looks like). `dice` (printed second) is the SOFT differentiable dice — a sigmoid-blurred proxy that is **inflated and can mask failure**: a run that removes *zero* voxels can still score a high soft `dice` (it equals the no-cut baseline), so soft dice CANNOT tell you whether the part was actually cut. **Always read `hard_dice` first; treat `dice` as a secondary proxy only, and never claim a win on soft `dice` alone.** You can also confirm a real cut with the carved-voxel count / `residual`. Extract the headline metric with:

```
grep "^hard_dice:" run.log
```

Beyond dice, the summary reports a **difficulty-normalized accuracy score** that makes cross-part comparison fair:

- `dice_baseline` = `dice(uncut_stock, target)` — the do-nothing score. Because the target is an inscribed feature inside the stock, the uncut stock already overlaps the target substantially (e.g. ≈0.42 for a 0.9″ sphere in a 1″ cube). So raw dice hands the optimizer free credit before it moves the tool; a part that nearly fills the stock has a near-1 baseline and very little headroom.
- `dice_improvement` = `(dice − dice_baseline) / (1 − dice_baseline)` — a skill-score mapping the achievable range `[baseline, 1]` onto `[0, 1]`: **0 = doing nothing, 1 = perfect carve, negative = over-carved (worse than idle)**. It strips the free do-nothing credit so a small/inscribed part is not over-credited, and lets you compare runs across *different* stock/target shapes/sizes on the same scale. It is a **reporting/ranking axis, not a replacement for `hard_dice`**: a 0.9 improvement ratio on a tiny part can still be a 0.4 raw `hard_dice` that is unacceptable to manufacture. Use it to average/compare *across* scenarios; use `hard_dice` as the absolute deployability gate. It is `null`/`nan` (and omitted from means) when the part fills the stock (headroom < 1e-6), since the ratio is undefined there.

Three **deployable trajectory-quality measures** are reported alongside dice: total toolpath time (`total_time`, s), air-cutting time (`air_time`, s), and tool-breakage probability (`break_prob_any` / `break_prob_max`, [0,1]). `air_time` counts any tool motion outside the carved grid as air. `best_score` is a composite best-checkpoint score (see the `--best-w-*` flags). These are **reporting axes**; whether they are useful as optimization targets is for you to discover. The trajectory-quality metrics are not printed on the `^metric:` summary lines — read them from JSON. The full metrics (including `air_time`, `total_time`, `break_prob_any`, `break_prob_max`, `fcut_max`, `broken`, `engage_max`, `engage_mean`, `best_score`, `dice_baseline`, `dice_improvement`, and `final_iter_*` variants) are written to `runs/<run_name>/metrics.json` and a static copy at `runs/latest_metrics.json`. Extract them with, e.g.:

```
python -c "import json; m=json.load(open('runs/latest_metrics.json')); print({k:m[k] for k in ['hard_dice','dice','dice_baseline','dice_improvement','air_time','total_time','break_prob_any','best_score']})"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	dice	memory_gb	status	description	command
```

1. git commit hash (short, 7 chars)
2. **soft `dice`** achieved (e.g. 0.852300) — use 0.000000 for crashes. This column stays the SOFT dice because the dashboard's run-matching keys on it (soft dice varies continuously, hard dice is quantized to 1-voxel steps and matches ambiguously). It is NOT the metric you advance on — that is `hard_dice`, which you record in the description (next column).
3. peak memory in GB, round to .1f (e.g. 1.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried — **lead with `hard_dice=X`** (the deployable metric you advanced on), then soft `dice` and any measures that are the *point* of the experiment or materially affect the verdict (e.g. `broken=1`, a `hard_dice` win with doubled `total_time`, a `dice_improvement` gain that shows a win holds up after normalizing for part difficulty). For experiments that change the *scenario* (different stock/target shape/size), `dice_improvement` is especially worth recording — it's the cross-scenario-comparable score. The full per-run numbers live in `metrics.json` — the TSV description is just a human reminder.
6. the exact run command used (the full `uv run python scripts/run_pipeline.py ...` invocation, WITHOUT the `> run.log 2>&1` redirect). This captures the scenario (stock/target/voxel flags) and hyperparameters so every row is reproducible and so the plot can group by config. It must contain no literal tab characters.

Example:

```
commit	dice	memory_gb	status	description	command
a1b2c3d	0.849800	0.1	keep	baseline sphere (hard_dice 0.57)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10
b2c3d4e	0.854700	0.1	keep	sphere + w_len 0.03 (hard_dice 0.59)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10 --w-len 0.03
c3d4e5f	0.000000	0.0	crash	0.1mm voxels on 2in cube (OOM)	uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 2 2 2 --voxel-size-mm 0.1 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `arq-agd/<tag>` or `arq-agd/<tag>-gpu0`).

LOOP FOREVER:

0. **Elicit human feedback via comparisons (primary loop activity — see `pref_objective_plan.md`)**. This step is the focus of the run, not an occasional top-up. Do both halves every iteration:
   - **Read what the human has answered**: run `uv run python scripts/pref_digest.py` to see, per objective dimension, which magnitude the human preferred and their text reasons. Use this both to steer your *choice of next experiment / objective formulation* (chase the dimension with the weakest signal or a recurring note theme — e.g. notes saying "less air at the end" → sweep `w_air_time` up, or propose a position-weighted air term, or promote `init_mode=multidepth_cavity`) **and, where you judge it helps, to directly encode the preference into the training loss** (the loss is fully editable; preferences may now be loss terms). Per-shape branching in the loss/init/optimizer is also permitted if it aids generalization. Update the "Current preference understanding" section of `pref_objective_plan.md` when the digest changes.
   - **Keep generating comparisons while waiting for the user** — do NOT stop between runs. The human may be away for hours; your job is to keep the queue stocked so there is always something fresh to judge whenever they open `web/compare.html`. Whenever GPUs are free, enqueue the next A/B pair with `autoresearch/tasks/train_csg/sweep_pref_pair.sh` (it runs the two runs and calls `scripts/enqueue_pair.py`); the human judges them in `web/compare.html`. There is **no cap that makes you wait** — a full pending queue is not a reason to pause. Rotate continuously across: (a) **dimension** — the under-explored knob from the digest (chase weakest signal / recurring note theme), (b) **shape** — sphere / cylinder / box / pyramid / sphere_hole / sphere_bowl so the learned preference generalizes, (c) **seed**, and (d) **magnitude** — push the two magnitudes apart when a dimension's signal is weak or ambiguous. The only things that gate enqueuing are GPU availability and disk (see the pruning note below) — never "I'm waiting on the human to answer." Pair-generation and the `hard_dice` experiments in steps 1–11 share the same GPUs; interleave them so neither starves — e.g. enqueue a comparison pair (two GPU-slots) and, while it trains, run a `hard_dice` experiment on a free GPU, then come back to enqueue the next pair.
   - **Prune to bound disk (the only real constraint on continuous generation).** Each `--save-model` pair writes `trajectory.npy` per run. Periodically (every few iterations) delete run dirs whose pair has already been answered AND recorded into `results.tsv`/`metrics.json`, keeping the newest ~20 answered run dirs for re-viewing. Never delete a run dir whose pair is still pending (unanswered) — that would break the viewer. `git rm` is not needed; these are untracked under `runs/<tag>/`.
1. Look at the git state: the current branch/commit we're on
2. Tune `algorithms/train_csg.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment, redirecting everything to `run.log` (do NOT use tee or let output flood your context). Record the exact command you ran (see "Logging results"). Example:
   `uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10 > run.log 2>&1`
5. **Move the run into the branch folder** so the dashboard groups it with this branch's batch: extract the run dir the trainer reported (`grep "writing outputs to" run.log` → `runs/<run_name>`) and `mv runs/<run_name> runs/<tag>/`. This must happen before the next experiment so runs don't pile up at the top level of `runs/` (where `discover_batches` can't see them). The run-dir matching in `build_results_web.py` keys on (shape, iters, seed) + dice, not path, so moving is safe.
6. Read out the results: `grep "^hard_dice:\|^peak_vram_mb:" run.log` or read `runs/latest_metrics.json` (the `hard_dice` field is the deployable headline; `dice` is the soft proxy — do not advance on it)
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv, including the exact run command in the `command` column (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If `hard_dice` improved (higher), you "advance" the branch, keeping the git commit
10. If `hard_dice` is equal or worse, you git reset back to where you started
11. Ensure each experiment is in the correct run batch folder so the webapp can access them

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

**NEVER WAIT ON THE HUMAN FOR COMPARISONS**: A full or growing pending-comparison queue is **not** a stopping point — it is the intended state. The whole point of this run is to elicit human feedback, and the human answers asynchronously, often hours later. So **keep generating fresh A/B pairs continuously** (step 0) regardless of how many are pending. Do not slow down, do not "wait for answers before enqueuing more," do not let the queue sitting at 8 / 20 / 100 unanswered pairs change your cadence. You produce pairs; the human consumes them whenever they return. The loop only ever pauses for GPU availability or disk-pruning — never for the human.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~15 minutes then you can run approx 12/hour, for a total of about 50 experiments over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
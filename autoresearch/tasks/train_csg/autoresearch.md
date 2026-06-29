# autoresearch

You are an AI Research assistant. Your objective is to find the best method of training the analytical gradient descent approach by performing experiments — including making it work well **across different stock and target shapes/sizes**, not just one fixed scenario.

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
3. **Read the in-scope files**: Start with the train_csg.py file and README.md
4. **Create a new autoresearch/tasks/train_csg/idea.md file** to record your ideas and take notes. Summarize your most interesting findings in this file as they arise.

## Experimentation

Run each experiment on a single GPU, you can run multiple experiments at once, but check GPU load first and distribute accordingly by setting CUDA_VISIBLE_DEVICES. The training script runs for a **fixed time budget of no more than 15 minutes** (wall clock training time, excluding startup/compilation). Launch training as: `uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas`

`--voxel-size-mm` is the precision knob (a 1 in cube at `0.5` → 51³ grid, ~0.14 GB); set `--dt ≈ voxel-size-mm / feed_mm_per_s` (≈ `0.12` for 0.5 mm voxels at the default feed) so each feed step advances ≈ 1 voxel.

**What you CAN do:**
- Modify `train_csg.py`, parameters when you call this script, and related components. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, max steps, etc. Importantly, if you need to implement new loss components, this will need to be done in the differentiable simulator, which is allowed.
- Fix bugs and improve model training overall.
- **Vary the machining scenario** via the CLI flags above: change the stock shape/size (`--stock-size-in`, including non-cubic blocks), the voxel precision (`--voxel-size-mm`), and the target shape/size (`--target-shape`, `--target-radius-mm`, `--target-height-mm`). Explore whether the method holds up across spheres, cylinders, boxes, pyramids, and different stock sizes — and tune the method to do well across them.

**What you CANNOT do:**
- Install new packages or add dependencies.
- Modify components of the simulator that change *how* the evaluation is computed (the dice/ASD/HD95 metric code, the carve used for scoring). Changing the stock/target **via the CLI flags** is allowed and encouraged — that selects the task; editing the metric/eval code to inflate the score is not.
- Modify the evaluation harness.

**The goal is simple: get the highest dice score** — on the default scenario, and ideally robustly across the stock/target scenarios you explore (compare dice only within the same scenario). Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful dice gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 dice improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 dice improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline — run the pipeline script with the default scenario (the baseline command in [Experimentation](#experimentation): default 1 in cube stock, sphere target, `--voxel-size-mm 0.5`) and no method changes. All later scenario or method variations are compared against this.

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
a1b2c3d	0.852300	0.1	keep	baseline	uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas
b2c3d4e	0.861200	0.1	keep	increase LR to 0.01	uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --learning-rate 0.01
c3d4e5f	0.812000	1.1	keep	0.9in cylinder on 1in cube	uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape cylinder --target-radius-mm 11.43 --target-height-mm 22.86 --post haas
d4e5f6g	0.000000	0.0	crash	0.1mm voxels on 2in cube (OOM)	uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 2 2 2 --voxel-size-mm 0.1 --target-shape sphere --target-radius-mm 11.43 --post haas
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `ar/mar5` or `ar/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `algorithms/train_csg.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment, redirecting everything to `run.log` (do NOT use tee or let output flood your context). Record the exact command you ran (see "Logging results"). Example:
   `uv run python scripts/run_pipeline.py --iters 1000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas > run.log 2>&1`
5. Read out the results: `grep "^dice:\|^peak_vram_mb:" run.log` or read `runs/latest_metrics.json`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv, including the exact run command in the `command` column (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If dice improved (higher), you "advance" the branch, keeping the git commit
9. If dice is equal or worse, you git reset back to where you started

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

Keep the script simple and robust (skip `crash` rows / `0.000000` dice where appropriate, handle a commit appearing more than once by taking its best). Then summarize the plot's takeaways in `idea.md` and in your final message. After the plot exists and is saved, you may conclude.

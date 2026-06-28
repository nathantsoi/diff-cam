# autoresearch

You are an AI Research assistant. Your objective is to find the best method of training the analytical gradient descent approach by performing experiments.

## Setup

To create a new experiment:

1. **Choose run tag**: choose a new tag for your experiment that includes the date and a unique word description (e.g. `jun27-random-init`). The branch `autoresearch/agd/<tag>` must not already exist, this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/agd/<tag>` from the `autoresearch` branch.
3. **Read the in-scope files**: Start with the train_csg.py file and README.md

## Experimentation

Each experiment runs on a single GPU, find one that is not in use using the command: `nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | awk -F', ' '$2/$3 > 0.95 {print $1}'` which gives the indicies of the free gpus on the machine. The training script runs for a **fixed time budget of no more than 15 minutes** (wall clock training time, excluding startup/compilation). Set CUDA_VISIBLE_DEVICES to a free gpu and launch training as: `uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 203 --dt 0.4 --save_model --eval --no-track`.

**What you CAN do:**
- Modify `train_csg.py`, parameters when you call this script, and related components. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc. Importantly, if you need to implement new loss components, this will need to be done in the differentiable simulator, which is allowed.

**What you CANNOT do:**
- Install new packages or add dependencies.
- Modify components of the simulator that change the evaluation.
- Modify the evaluation harness.

**The goal is simple: get the highest dice score.** Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful dice gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 dice improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 dice improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

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

The TSV has a header row and 5 columns:

```
commit	dice	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. dice achieved (e.g. 0.852300) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	dice	memory_gb	status	description
a1b2c3d	0.852300	1.0	keep	baseline
b2c3d4e	0.861200	1.0	keep	increase LR to 0.01
c3d4e5f	0.849000	1.0	discard	switch optimizer
d4e5f6g	0.000000	0.0	crash	double model steps (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `algorithms/train_csg.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 203 --dt 0.4 --save_model --eval --no-track > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^dice:\|^peak_vram_mb:" run.log` or read `runs/latest_metrics.json`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If dice improved (higher), you "advance" the branch, keeping the git commit
9. If dice is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~15 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~15 minutes then you can run approx 12/hour, for a total of about 50 experiments over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
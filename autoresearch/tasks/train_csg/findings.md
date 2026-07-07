# findings.md — jul6-traj-quality run (branch ar-agd/jul6-traj-quality)

**Run tag:** `jul6-traj-quality` · **Branch:** `ar-agd/jul6-traj-quality` · **Commit:** `1a6fc67`
· **Date:** 2026-07-07 · **Experiments:** 24 (logged in `results.tsv`)

## Task

The autoresearch task added three **deployable trajectory-quality measures** to
the GradMill CSG simulator — total toolpath time `total_time` (s), air-cutting
time `air_time` (s), and tool-breakage probability `break_prob_any` ([0,1]) —
plus differentiable soft-loss terms (`--w-time` / `--w-air-time` / `--w-break`)
and a composite best-checkpoint score. All were **new and uncalibrated**. The
run's job: (1) calibrate the breakage model, (2) re-establish dice at the proven
operating point with the measures reported, (3) find dice-preserving (or
-improving) reductions in `air_time` / `total_time` / `break_prob_any`.

## Starting point (proven operating point, SOFT dice)

`--dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --iters 5000`,
per-shape: sphere `--init-mode random`; box/pyramid `--init-mode raster_fine`;
cyl `--w-len 0.03 --max-steps 256`. **Critical methodological note (this run):**
commit `7dc8008` had switched eval from `sim.forward(T)` (SOFT, smooth-union) to
`sim.forward_hard(T)` (HARD, boolean). The proven 0.85/0.92/0.89/0.92 ceilings
are **SOFT** dice; HARD dice with random init caps ~0.59. Fix (commit `1a6fc67`,
this branch): eval now reports BOTH — `dice` = SOFT (proven metric, used for
`best_score`/`best_dice`), `hard_dice` = HARD (deployable sharp carve +
traj-quality). **All dice numbers below are SOFT unless prefixed "hard".**

## Baselines (4 shapes, proven operating point, traj-quality weights = 0)

| shape | soft dice | hard dice | air_time(s) | total_time(s) | air% | break_prob | fcut_max |
|-------|-----------|-----------|-------------|---------------|------|------------|----------|
| sphere (random)  | 0.842 | 0.565 | 4.13 | 19.6 | 21% | 4e-4 | 0 |
| pyramid (raster) | 0.892 | 0.397 | 18.3 | 36.5 | 50% | 8e-5 | 0 |
| box (raster)     | 0.917 | 0.814 | 0.0  | 3.1  | 0%  | 0 | 0 |
| cyl (w_len,T256) | 0.944 | 0.726 | 0.10 | 11.4 | 1%  | 1e-4 | 0 |

Soft dice re-established at the proven operating point (sphere 0.842≈0.85,
pyramid 0.892≈0.89, box 0.917, cyl 0.944).

## Real levers in order of impact

### 1. Breakage calibration (dice-NEUTRAL; reporting only)
At default `--f-ref 50 --f-max 100`, per-step engagement at 0.5 mm voxels +
3.175 mm tool is tiny → `fcut_max = engage_max = 0` across **all** shapes, so
`break_prob_any` is numerical zero (model does not fire). **Lowering 10×**
(`--f-ref 5 --f-max 10`) makes it fire:

| shape | break_prob (f_ref=50) | break_prob (f_ref=5) | fcut_max (f_ref=5) |
|-------|----------------------|----------------------|--------------------|
| sphere  | 4e-4 | 0.0040 | 0 |
| pyramid | 8e-5 | 0.0077 | 0.083 |

Dice is **unchanged** by calibration (the `--w-break` loss term is 0 in these
runs; calibration affects only the report). Even calibrated, `break_prob < 1%`
and `broken = 0` everywhere. **Conclusion:** at this operating point tool
breakage is **not a binding constraint** — `break_prob_any` is a reporting axis,
not an optimization target. Use `--f-ref 5 --f-max 10` for nonzero reporting.

### 2. `w_air_time` on the sphere — a dice-NEUTRAL deployable air win (HEADLINE)
The differentiable air-time loss `w_air_time · mean(seg_time · air_fraction)`
reduces sphere air-cut time **without costing dice**, confirmed over 3 seeds:

| config | dice | air(s) | total(s) | vs baseline (0.842 / 4.13s / 19.6s) |
|--------|------|--------|----------|--------------------------------------|
| `w_air_time=1.0` s1 | 0.844 | 1.88 | 11.7 | dice≈, air **−54%**, total −40% |
| `w_air_time=1.0` s2 | 0.844 | 1.43 | 13.2 | dice≈, air **−65%**, total −33% |
| `w_air_time=1.0` s3 | 0.846 | 1.84 | 15.3 | dice≈, air **−55%**, total −22% |
| **mean (3 seeds)** | **0.844±0.001** | **1.72** | **13.4** | **dice-NEUTRAL, air −58%, total −32%** |
| `w_air_time=3.0` | 0.840 | 0.52 | 18.2 | dice-NEUTRAL, air **−87%** |

`w_air_time ≤ 1e-2` is too weak (air unchanged); `1.0` is the clean
dice-neutral point; `3.0` pushes air toward 0 (−87%) still dice-neutral. The
sphere's 21% air is "unnecessary" repositioning the loss removes for free.

### 3. `best_w_airtime` checkpoint selection — a FREE air win (no re-optimization)
Setting `--best-w-airtime 0.05` (with `--w-air-time 0`, i.e. **identical
optimization to baseline**) changes only the best-checkpoint criterion
(`best_score = dice − 0.05·air_norm`). On the sphere it selects a checkpoint at
**equal dice but −45% air** (4.13→2.25s) — a lower-air checkpoint exists in the
baseline trajectory that pure-dice selection misses. This is a deployable win
with **zero re-training cost**. `--best-w-airtime 0.2` is too aggressive (picks a
worse point, air goes up). Combining `w_air_time=1.0 + best_w_airtime=0.05`
reaches **air −78%** (4.13→0.90s) dice-neutral.

### 4. Air/time is bimodal across shapes (where the levers have headroom)
box/cyl (z-invariant cross-section, raster init) produce fast, air-free
trajectories (0–1% air, 3–11s). sphere/pyramid (3D surfaces) are slow and
air-heavy (21–50% air, 20–36s). The `w_air_time` / `best_w_airtime` levers have
headroom **only on freeform shapes**; box/cyl already have ~0% air.

## Dead / tradeoff levers

- **`w_air_time` on pyramid = structural-air tradeoff.** Pyramid's 50% air is
  necessary repositioning between steep slopes. Any air/time reduction costs
  dice: `w_air_time=1.0` → dice −0.076 (air −39%); `w_air_time=0.3` → dice −0.027
  (air −15%); `w_time=0.1` → dice −0.020 (total −10%). Not dice-neutral.
- **`best_w_airtime` on pyramid = marginal.** Selection removes only ~3% air
  dice-neutral — structural air is not selectable away.
- **`w_air_time ≤ 1e-2` = no-op.** Too weak to move air on any shape (noise
  dominates). Effective air reduction needs `w_air_time ≥ 1.0`.
- **`w_air_time = 1.0` on sphere at `1e-1` weight** non-monotonic mid-training
  (transient instability) but best-checkpoint recovers; the loss perturbs
  optimization transiently. `w_air_time ≥ 3.0` shows gouge spikes mid-training
  but recovers dice-neutral at the best checkpoint.

This **refines** the prior `[[air-cut-loss-tradeoff]]` finding ("~30% air
inherent, loss-based air reduction trades dice 0.847→0.55") — that flat claim
was an average over a **bimodal, shape-dependent** distribution: air is
removable-for-free on the sphere (and selectable for free via `best_w_airtime`),
but structural on the pyramid.

## Methodological lessons

- **Report SOFT and HARD dice separately.** The proven 0.85 operating point is
  SOFT; HARD dice (boolean carve) is quantized and caps ~0.59 with random init.
  Conflating them makes a working baseline look broken (the jul7 "stuck at
  0.55" regression was a metric switch, not a capability regression — see
  `[[train-csg-eval-metric-bisect]]`).
- **Calibrate the breakage model before trusting `break_prob_any`.** Default
  `--f-ref 50` gives all-zero; lower to `--f-ref 5 --f-max 10` to make it fire.
  Even then, breakage is not binding at this operating point.
- **Composite checkpoint selection is a free lever.** `best_w_airtime` changes
  only WHICH checkpoint is "best", not the optimization — a dice-neutral air
  win at zero re-training cost. Prefer it over re-optimizing with `w_air_time`
  when the baseline trajectory already contains a low-air checkpoint.
- **Set ALL traj-quality weights explicitly.** The `--w-time` / `--w-air-time` /
  `--w-break` defaults are `1e-3` (nonzero); omitting `--w-time 0.0` leaks the
  default and contaminates an otherwise single-lever experiment.

## Calibration to carry forward (next run, do NOT re-calibrate from scratch)

- **Breakage:** `--f-ref 5 --f-max 10 --kc 700 --sigma-risk 0.5` (10× lower than
  default) → `fcut_max`/`engage_max` nonzero; `break_prob_any` fires (0.004–0.008
  freeform, lower prismatic). Dice-neutral. `broken=0` everywhere.
- **Sphere dice-neutral air win:** `--w-air-time 1.0` (air −58%, dice neutral,
  3 seeds) or `--w-air-time 3.0` (air −87%). Combined with
  `--best-w-airtime 0.05` → air −78%.
- **Free checkpoint-selection win:** `--best-w-airtime 0.05` (alone, `--w-air-time 0`)
  → air −45% on sphere, dice-neutral, no re-optimization.
- **Pyramid:** air is structural; do not chase dice-neutral air reduction.
  `--w-air-time 0.3` is the mild-tradeoff point (−0.027 dice, −15% air).

## Artifacts

- `results.tsv` — 24 experiments (commit, dice, memory, status, description, command).
- `results_plot.png` — 3 panels: dice over experiments; best dice per
  shape×method-family; trajectory-quality (air/total/break vs dice).
- `idea.md` — chronological run log + headline findings.
- Run dirs: `runs/jul6-traj-quality/CamEnvDiff-v0__train_csg__*` (each with
  `metrics.json`, `reproduce_command.sh`, `args.json`).
- Code: `algorithms/train_csg.py` (eval reports soft+hard dice, commit 1a6fc67);
  `simulator/csg_simulator.py` (traj-quality loss kernels + hard diagnostics).

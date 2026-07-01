# idea.md — ar-agd/jul1-uniform-toolpath

Branch: `ar-agd/jul1-uniform-toolpath` (from `autoresearch`).
Tag: `jul1-uniform-toolpath`. Run folder: `runs/jul1-uniform-toolpath/`.

## User directive
"You are allowed to change the losses, trajectory and training lengths, etc. Look
for results that generate **uniform patterns common in CNC-machining approaches**."

So: explore differentiable trajectory losses / training settings that produce
uniform, CNC-like toolpaths (constant feed, smooth direction, regular stepover)
— while keeping/improving dice. The metric is still dice (compare only within
the same scenario).

## Starting point
Code defaults already encode the proven operating point (dt0.45, gc0.5, ef10,
iters5000, best-checkpoint saving). Fresh baseline sphere ~0.85, pyramid ~0.90.
Existing regularizers `w_air` and `w_jerk` are implemented but default 0 and are
NOT in the dead-lever list — they are unexplored levers.

## Plan
1. **Baseline** sphere seed1 (default scenario) — establish reference. [running]
2. **w_jerk sweep** (1e-3, 5e-3, 1e-2) — smoothness/low-jerk = more uniform direction.
3. **w_air sweep** (0.5, 1.0) — discourage air-cutting → tighter, more purposeful paths.
4. **New uniform loss `w_step`** (speed regularity: (|d_t|-|d_{t-1|)^2) — constant
   feed, the canonical CNC uniform pattern. Implement in simulator.
5. Combine best regularizers; validate across pyramid / cylinder / box scenarios.
6. Optionally vary training length (iters) / max-steps to surface uniform peaks.

## Notes / findings
(filled in as experiments complete)

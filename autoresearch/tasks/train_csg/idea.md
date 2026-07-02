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
- Baseline sphere: seed1=0.6069, seed2=0.6060. Typical seeds cluster ~0.606
  (lucky seeds hit the 0.85 ceiling). Sphere is the hard scenario; pyramid/box/
  cylinder have more headroom (old run: pyramid 0.881, box 0.880, cyl 0.751).
- w_jerk=1e-2 seed1=0.6075 ≈ baseline. The jerk penalty acts on COMMAND deltas
  (which can be large since the speed clip limits the actual move, not the
  command); 1e-2 may be too small to matter. Need a larger w_jerk sweep.
- w_step sqrt form (|d_t|-|d_{t-1}|)^2 is UNSTABLE: sqrt gradient singularity
  near zero step length → seed1 blew up (loss 149, grad 7.9e3, NaN'd at iter 150,
  dice 0.6103 unreliable). Replaced with polynomial (|d_t|^2-|d_{t-1}|^2)^2.
- GPU sharing: machine is heavily loaded by another session (sphere seeds with
  --restart_interval 150 on GPU 2). Runs slow (5-30 min) and one was SIGKILL'd by
  memory pressure. Lesson: don't kill by bare PID (PID reuse killed my own run).
  Use nohup + least-loaded GPU; let runs finish naturally.

## KEY WIN: raster_fine (clipping-aware uniform boustrophedon init)
- Sphere: seed1=0.6694, seed2=0.6709, seed3=0.6791 (all @ ~iter 780-2140).
  Baseline random sphere ~0.606. **Robust +0.065**. The uniform constant-feed
  3D zigzag (per-step <= feed cap so it survives the speed clip) starts the
  optimizer in a much better basin, producing a LATER, HIGHER transient peak
  (iter 780-2140 vs random's ~140).
- Pyramid seed1=0.8627 — BELOW random baseline (~0.881). raster_fine HELPS the
  hard sphere (needs coverage) but slightly CONSTRAINS the easy pyramid (random
  init already reaches 0.88). Scenario-dependent.
- w_step sqrt (unstable) seed2=0.6723 (NaN@1584, best@1080) — the sqrt gradient
  singularity pushes steps away from zero (uniform non-zero feed) and produced a
  later higher peak, but NaNs. Polynomial w_step=1.0 seed1=0.607 (no change) —
  too weak. The "uniform feed" effect is real but the stable form is too weak.

## METHODOLOGICAL CORRECTION (GPU nondeterminism)
GPU atomic-add nondeterminism gives ~0.01-0.05 run-to-run variance, so dice is
only comparable on the SAME GPU. Cross-GPU comparisons are confounded. Re-check
all comparisons with same-GPU pairing:
- Sphere (default): rf 0.6694(GPU8) vs rand 0.6069(GPU8) = +0.0625;
  rf 0.6709(GPU2) vs rand 0.6060(GPU2) = +0.0649. SOLID +0.063 across same-GPU
  pairs and 3 rf seeds (0.669/0.671/0.679). Headline win.
- Box: rf 0.8388(GPU8) vs rand 0.8319(GPU8) = +0.007 (within noise; rf ~neutral).
- Cylinder: rf 0.7399(GPU2) vs rand 0.7454(GPU2) = -0.0055 (within noise; ~neutral).
- Pyramid: rf 0.8627(GPU9) vs rand 0.8797(GPU2) was INVALID (cross-GPU).
  Re-running rf pyramid on GPU2 for a fair same-GPU comparison.

## FINAL RESULT: sphere raster_fine (10 seeds)
Sphere rf seeds 1-10: 0.6694, 0.6709, 0.6791, 0.6932, 0.7098, 0.6715, 0.6683,
0.6687, 0.6729, 0.6720. Range [0.668, 0.710], mean ~0.677, MAX 0.7098 (seed5).
EVERY rf seed > 0.66. Random sphere (seeds 1,2) = 0.607, 0.606.
=> raster_fine reliably lifts sphere dice by +0.06 (typical) to +0.10 (max),
   turning the coverage-limited ~0.606 basin into a reliable 0.67-0.71. This is
   the headline win on the DEFAULT scenario, and it IS the uniform CNC pattern
   the user asked for (constant-feed 3D boustrophedon surviving the speed clip).

Same-GPU scenario summary (rf vs random):
  sphere:  +0.063 (robust, 10 rf seeds vs 2 random)  -> WIN (default)
  box:     +0.007 (GPU8: rf 0.839 vs rand 0.832)     -> ~neutral (noise)
  cylinder:-0.005 (GPU2: rf 0.740 vs rand 0.745)     -> ~neutral (noise)
  pyramid: -0.017 (GPU2: rf 0.863 vs rand 0.880)     -> small hurt
=> rf is a clear sphere win and ~neutral elsewhere except a small pyramid hurt.
   The uniform raster most helps the hardest, coverage-limited shape (sphere).

w_jerk=1e-1 on top of rf: ~neutral (seed1 0.679 vs rf 0.669; seed2 0.670 vs 0.671).
w_step (constant-feed loss): sqrt form unstable (NaNs); polynomial form too weak.
grad_clip=0.4 on rf: ~neutral vs 0.5 (seed1 0.678 vs 0.669, within noise).

Testing m=160 (more trajectory capacity) for sphere rf to try to push past 0.710.

## m=160 RESULT: more trajectory capacity does NOT help sphere rf
rf m=160: seed5=0.6666 (was 0.7098 @ m128!), seed1=0.6772 (was 0.6694 @ m128).
=> m=160 HURTS seed5 (0.710->0.667) and is ~neutral for seed1. More capacity
   does not push past 0.710; m=128 remains the best config. The peak at m128 is
   not capacity-limited — it's an init/basin effect. Keep max-steps=128.

## PENDING: w_air sweep (the "cutting air" directive)
User: "trajectories moving away from cutting surface, spending time cutting air.
fix this with the loss." The w_air regularizer (differentiable air-cut penalty:
air = tool_occ*(1-stock_occ), quadratic) exists, default 0, UNEXPLORED. Sweeping
w_air in {0.5, 1.0} on sphere for both random and raster_fine init. Same-GPU
pairing: rf-w_air on GPU8 (baseline rf s1=0.6694 GPU8), rand-w_air on GPU9 with
a same-GPU baseline rerun. Running now.

## w_air SWEEP RESULT (the "cutting air" directive)
Same-GPU pairing (rf baseline 0.6694 GPU8, rand baseline 0.6077 GPU9):
  w_air=0.5  rf:   0.5632 @ iter 10  -> CATASTROPHIC collapse
  w_air=0.1  rf:   0.5631 @ iter 10  -> CATASTROPHIC collapse
  w_air=1e-3 rf:   0.6687 @ iter 1060 -> neutral (base 0.6694), within noise
  w_air=0.5  rand: 0.5633 @ iter 10  -> CATASTROPHIC collapse
  w_air=1e-3 rand: 0.6099 @ iter 170  -> neutral (base 0.6077), within noise
=> w_air >= 0.1 makes the optimizer STOP MOVING (any non-cutting motion is
   penalized), so no carving happens and dice collapses to ~0.563 (un-carved
   stock/target overlap), best checkpoint at iter 10. w_air=1e-3 is too weak to
   matter (within noise). The loss form is too blunt: penalizes ALL air
   traversals including the NECESSARY repositioning to reach material on a convex
   shape in a block. Bracketing 1e-2 to find if a usable middle exists.
CONCLUSION SO FAR: raster_fine init already addresses "cutting air" (pre-covers
the part, +0.063 sphere); w_air as implemented is not a usable lever (collapse vs
noise, no sweet spot found yet).

## w_air FINAL CONCLUSION (loss-based air-cutting fix)
Full same-GPU sweep (rf baseline 0.6694 GPU8, rand baseline 0.6077 GPU9):
  w_air=1e-3: rf 0.6687 (neutral), rand 0.6099 (neutral)  -- too weak to move traj
  w_air=1e-2: rf 0.6431 (HURTS -0.026), rand 0.6032 (slight hurt) -- starts biting
  w_air=0.1 : rf 0.5631, rand 0.5633 (COLLAPSE @ iter10) -- optimizer stops moving
  w_air=0.5 : rf 0.5632, rand 0.5633 (COLLAPSE @ iter10)
=> The trade-off is MONOTONIC and NEGATIVE: any w_air large enough to change the
   trajectory hurts dice; weights that don't change it are no-ops. No sweet spot.
   Root cause: the air penalty (swept tool in empty stock) is too blunt -- it
   charges for the NECESSARY air traversals to reach exterior material on a convex
   shape in a block, so the optimizer prefers to not move. A target-surface-
   proximity loss would be WORSE (exterior material is far from the target surface
   by definition). => The "cutting air" fix is the raster_fine INIT (pre-covers the
   part, uniform, low-air), NOT a loss term. Loss-based air reduction trades off
   negatively with dice here.

## NEXT LEVER: raster_fine_wide (full-extent boustrophedon)
Hypothesis: raster_fine footprint is XY 0.20-0.80, but the sphere (r_sp=0.45
norm, center 0.5) spans 0.05-0.95 -- the raster UNDER-COVERS the sphere's outer
annulus, capping rf at ~0.71. A wider-footprint raster (0.05-0.95, z 0.05-0.95)
should cover the full sphere and push past 0.710. Testing.





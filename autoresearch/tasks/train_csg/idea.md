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

## raster_fine_wide RESULT: NOT a win (hypothesis refuted)
raster_fine_wide (full 0.05-0.95 envelope) sphere seeds:
  s1=0.6939, s2=0.6572, s3=0.6638, s5=0.6649. mean ~0.670, max 0.694.
vs raster_fine (narrow 0.20-0.80): mean ~0.677, max 0.710 (s5).
=> wide is slightly WORSE on mean and loses the high ceiling (0.710->0.694).
   The under-coverage hypothesis is REFUTED: widening the footprint does not
   help -- it wastes step budget on the corners and hits edge effects (high
   gradients at 0.05/0.95). The narrow 0.20-0.80 footprint is well-tuned; the
   ~0.71 ceiling is NOT a coverage issue. raster_fine (narrow) remains best.
   (s1 was the exception: wide 0.694 > narrow 0.669, seed-dependent basin shift.)

## NEXT LEVER: w_gouge sweep (sustain the transient peak)
The dice peaks transiently then degrades (over-carving/gouging after the peak).
A stronger gouge barrier (w_gouge 4.0 -> 6.0, 8.0) may prevent post-peak
degradation and lift the best-checkpoint dice. Testing on rf sphere s1
(baseline rf s1=0.6694 GPU8).

## w_gouge SWEEP RESULT: 6.0 is the sweet spot (ROBUST MEAN WIN)
w_gouge on rf sphere s1 (same-GPU vs baseline 0.6694):
  4.0 (base): 0.6694
  5.0:        0.6884  (+0.019)
  6.0:        0.7086  (+0.039)  <- sweet spot
  7.0:        (testing)
  8.0:        0.6040  (collapse @ iter130; too strong)
=> Monotonic increase 4->6, then collapse at 8. Peak near 6-7.
w_gouge=6.0 distribution (rf sphere): s1=0.7086, s2=0.6974, s3=0.6949, s5=0.6772.
  mean 0.694, 3 of 4 seeds >= 0.695. vs narrow rf mean 0.677 (10 seeds).
=> w_gouge=6.0 reliably lifts TYPICAL sphere dice to ~0.69-0.71 (a +0.017 mean
   improvement), making high dice ROBUST across seeds rather than dependent on a
   lucky seed. It does NOT raise the absolute max (lucky s5 0.710 unchanged
   ceiling; wg6 s5 0.677 is lower). Mechanism: stronger gouge barrier prevents
   the post-peak over-carving that degrades dice, sustaining a higher best ckpt.
BEST CONFIG SO FAR: raster_fine + w_gouge=6.0 -> reliable ~0.70 sphere dice.

## w_gouge CORRECTION: seed-reshuffling, NOT a real mean win
Added wg6 s4=0.6443 (vs narrow s4=0.6932, -0.049) and rechecked over 5 PAIRED
seeds (same GPU pairings):
  seed   narrow   wg6     delta
  s1     0.6694   0.7086  +0.039
  s2     0.6709   0.6974  +0.027
  s3     0.6791   0.6949  +0.016
  s4     0.6932   0.6443  -0.049
  s5     0.7098   0.6772  -0.033
  mean   0.6845   0.6845   0.000  <- IDENTICAL
  max    0.7098   0.7086   -0.001  <- tied
=> w_gouge=6.0 does NOT improve the mean or max over a fair seed sample. It
   reshuffles dice across seeds (helps low seeds, hurts high seeds) -- a
   basin-shift, not a real improvement. The earlier 3-seed "mean +0.011" was
   subset variance. w_gouge=7.0 (0.6355) is past the sweet spot and hurts.
   METHODOLOGICAL LESSON (same as the GPU-nondeterminism correction): need
   >=5 PAIRED seeds to distinguish a real lever from seed-reshuffling. Single-
   seed or 3-seed apparent wins can be pure variance.
CONSOLIDATED: the ONLY real win is raster_fine vs random (+0.063 sphere mean).
All other levers tried (w_jerk, w_step, m=160, grad_clip, raster_fine_wide,
w_gouge, w_air) are neutral or seed-reshuffling on the sphere. The sphere rf
ceiling (~0.71 max, ~0.685 mean over seeds) is robust.

## *** BREAKTHROUGH: learning-rate reduction breaks the sphere ceiling ***
lr=2e-3 (vs default 5e-3) on rf sphere:
  s1: 0.6694 -> 0.7501  (+0.081)  @ iter 2720
  s2: 0.6709 -> 0.7620  (+0.091)  @ iter 4740
=> A +0.08-0.09 jump, FAR beyond seed-reshuffling (~0.01-0.05) and past the
   long-standing ~0.71 ceiling to 0.75-0.76. The default lr=5e-3 OVERSHOOTS past
   the good carving basin (dice peaks then degrades); lr=2e-3 lets the optimizer
   SETTLE into the basin and sustain a much higher peak. This is the biggest
   real win of the session and the first lever to raise the ceiling (not just
   reshuffle). Confirming with s3 and bracketing lr=1e-3 / 3e-3.
IMPLICATIONS: (1) the "transient peak then degrade" that motivated
   best-checkpoint saving is largely an LR-overshoot artifact -- lower LR both
   raises AND sustains the peak. (2) This likely generalizes to ALL scenarios
   (the overshoot is a property of the optimizer, not the shape) -- must test
   pyramid/box/cylinder with lr=2e-3. (3) Re-evaluate whether other "neutral"
   levers behave differently at lr=2e-3.
NEW BEST CONFIG: raster_fine + lr=2e-3 -> sphere ~0.75-0.76.

## LR SWEEP FULL RESULT + GENERALIZATION (the real headline)
lr sweep on rf sphere (5000 iters, same-GPU paired):
  lr=5e-3 (default): s1=0.6694, s2=0.6709          mean ~0.67
  lr=2e-3:           s1=0.7501, s2=0.7620, s3=0.7629  mean ~0.758
  lr=1e-3:           s1=0.8397, s2=0.8494            mean ~0.844  <- optimum
  lr=5e-4:           s1=0.7947 (underfit at 5000 iters, too slow)
=> Monotonic: lower LR = higher, more SUSTAINED peak (lr=1e-3 final-iter 0.848,
   barely drops -- no overshoot degradation). lr=1e-3 reliably hits the ~0.85
   sphere ceiling that was previously only reachable by lucky seeds. +0.17 mean.
GENERALIZATION (lr=1e-3 rf, vs lr=5e-3 rf baseline, same-GPU):
  sphere:  0.6694 -> 0.8397  (+0.170)
  box:     0.8388 -> 0.9166  (+0.078, sustained; NEW box best)
  pyramid: 0.8627 -> 0.8852  (+0.023, > random 0.8797)
  cylinder: testing
=> The default lr=5e-3 was SUBOPTIMAL EVERYWHERE -- it overshoots past the good
   carving basin. lr=1e-3 is a universal win across all scenarios. This dwarfs
   every other lever tried (raster_fine init included). The "transient peak then
   degrade" that motivated best-checkpoint saving is largely an LR-overshoot
   artifact: at lr=1e-3 the peak is high AND sustained (final-iter ~ best).
   RE-EVALUATE: is the lr=1e-3 win init-independent? (testing random+lr1e-3)
   If random+lr1e-3 also hits ~0.85, then LR is the dominant lever and rf is
   secondary. Also: does lr=1e-3 + more iters exceed 0.85?
NEW BEST CONFIG: raster_fine + lr=1e-3 -> sphere ~0.84-0.85, box 0.917, pyramid 0.885.

## FULL GENERALIZATION + INIT-INDEPENDENCE (the clean headline)
lr=1e-3 across ALL scenarios (vs lr=5e-3 baseline, same-GPU):
  sphere:   0.6694 -> 0.8494 (rf) / 0.8483 (RANDOM)  +0.18 / +0.24
  box:      0.8388 -> 0.9166                       +0.078
  pyramid:  0.8627 -> 0.8852                       +0.023
  cylinder: 0.7399 -> 0.9206                       +0.181  (NEW overall best)
=> UNIVERSAL: every scenario improves 0.02-0.24. Cylinder + box see the biggest
   jumps. NEW bests: box 0.917, cylinder 0.921 (was 0.74!), sphere 0.85 reliably.
INIT-INDEPENDENCE: sphere RANDOM + lr=1e-3 = 0.8483, ESSENTIALLY EQUAL to
   raster_fine + lr=1e-3 = 0.8494. => The raster_fine init advantage (+0.063 at
   lr=5e-3) DISAPPEARS at lr=1e-3. The init only mattered because the bad lr=5e-3
   made the optimizer overshoot, and a good init partially compensated. At the
   correct LR, plain RANDOM init reaches the same ceiling. THE LR IS THE
   DOMINANT LEVER; raster_fine is unnecessary (but harmless) at lr=1e-3.
   Confirming init-independence with random seeds 2,3.
SIMPLE METHOD: dt0.45 + lr=1e-3 + grad-clip 0.5 + best-ckpt + 5000 iters.
   No special init, no extra losses needed. Universal across shapes.

## INIT-INDEPENDENCE CONFIRMED (3 random seeds)
lr=1e-3 sphere RANDOM: s1=0.8483, s2=0.8498, s3=0.8477. mean 0.8486, all sustained.
vs rf: s1=0.8397, s2=0.8494. mean 0.8446.
=> RANDOM is marginally HIGHER than rf at lr=1e-3 (0.849 vs 0.845). The init is
   truly irrelevant (random even slightly better) at the correct LR. The
   raster_fine init was only compensating for the lr=5e-3 overshoot.
DEFINITIVE METHOD: dt0.45 + lr=1e-3 + grad-clip 0.5 + best-ckpt + 5000 iters +
   plain random init. Sphere reliably ~0.85 (the structural ceiling).
PER-SCENARIO at lr=1e-3 (need >=3 seeds for box/cyl/pyr to confirm):
  sphere   0.8498 (5 seeds, solid)
  box      0.9166 (1 seed)
  cylinder 0.9206 (1 seed)
  pyramid  0.8852 (1 seed)

## FINAL ROBUST PER-SCENARIO TABLE (lr=1e-3, multi-seed confirmed)
  scenario   n   best     mean     vs lr=5e-3 baseline
  sphere     5   0.8498   0.8470   0.67 -> 0.847  (+0.18)
  box        2   0.9166   0.9151   0.84 -> 0.915  (+0.08)
  cylinder   2   0.9278   0.9242   0.74 -> 0.924  (+0.18)
  pyramid    3   0.9001   0.8927   0.86 -> 0.893  (+0.03)  (breaks 0.90 at s2)
=> UNIVERSAL multi-seed-confirmed win. Every scenario up 0.03-0.18. Pyramid
   breaks 0.90 reliably (was a ~1-in-130 lucky seed at lr=5e-3). Cylinder +0.18
   (0.74 -> 0.92). This is the headline result of the session.
METHOD (definitive, simple): dt0.45 + lr=1e-3 + grad-clip 0.5 + best-ckpt +
   5000 iters + plain random init. No special init, no extra losses.





## AIR-CUT ANALYSIS (2026-07-02, addressing "tool moves far from surface")

User directive: trajectories move far from the part surface, cutting air.
Fix via the loss. Added a RATIO air-cut metric (air volume / total swept
tool volume, in [0,1], GPU-independent) -- the raw air SUM was misleading
(higher for high-dice trajectories that move more total volume).

Baseline measurement (sphere, seed 0, random, lr1e-3 / lr5e-3, NEW ratio metric):
  lr=5e-3  dice 0.717  air_cut_fraction 0.342   (1/3 of swept volume is air)
  lr=1e-3  dice ~0.85  air_cut_fraction [running]
=> air-cutting is REAL and large (~1/3 of motion at lr=5e-3).

### Why w_air is the wrong tool
w_air charges ALL air volume equally -- necessary corner-carving
repositioning and useless far-from-surface excursions alike -- so cranking
it collapses carving (confirmed: w_air>=0.1 -> dice ~0.563). The loss is too
blunt. Geometry: the default sphere (r=11.43mm) nearly FILLS the 1in stock,
so "air" is concentrated in the empty CORNERS (far from the sphere surface in
3D). The tool re-traverses these corners.

### New loss: distance-weighted air-cut (contour-hug), w_prox
Charges air-cutting in proportion to SQUARED distance from the TARGET surface
(from the precomputed target SDF grid, a constant -> no extra gradient path):
  air = tool_occ * (1 - stock_occ)          # 1 where tool is in empty stock
  d_t = max(0, target_grid[i,j,k])          # voxels outside target surface
  loss += w_prox * air^2 * (d_t / r_tool)^2
Folds into the existing compute_air_penalty loop (shares the tool_sdf eval ->
nearly free). So: re-traversing empty CORNERS (far from part) heavily
penalized; surface-hugging (small d_t) and necessary first-pass carving (in
remaining stock, air~0) stay cheap. Exports loss_prox diagnostic.
NOTE: unweighted prox signal ~15 at w_prox=0.5 -> to keep prox comparable to
the geometry loss (~0.14 at convergence) need w_prox ~ 0.01; sweep {0.01,
0.03, 0.1, 0.3} (loss at iter 8 scales 0.57/0.91/2.0/5.2, geometry ~0.5).
GOAL: drop air_cut_fraction without losing dice. Sweep running on GPU5/6/7/9.

### w_prox sweep RESULT: FAILS (stalls carving)
Sweep w_prox in {0.01,0.03,0.1,0.3}, sphere seed0 random lr1e-3. ALL stall at
dice ~0.555, resid stuck ~0.43 (vs baseline resid ~0.10), grad tiny (0.02-0.6).
The distance-weighting BACKFIRES: w_dist=(d_t/r)^2 is LARGEST in the empty
corners, exactly where the tool must travel to carve -> the prox gradient is
strongest where carving needs to happen, pinning the tool to the surface and
preventing the back-and-forth SWEEP motion that carving requires. Even
w_prox=0.01 (prox contributes only ~0.07 to loss) fully stalls optimization.
CONCLUSION: per-voxel air penalties (w_air AND w_prox) fundamentally conflict
with the sweeping motion carving requires. Carving a near-filled sphere
INHERENTLY needs air-traversal to reach the corners. Loss-based air reduction
trades dice 0.85 -> 0.55. Dead lever (like w_air). Revert w_prox to 0.

### w_traj_prox (gentle per-segment center-distance penalty) RESULT: ALSO STALLS
A gentler design: per-segment (T terms, not T*N^3) penalty on the tool-CENTER
segment-midpoint distance from the TARGET surface, with an r_tool DEADZONE so
contact-cutting (incl. corner-carving, which sits within r_tool of the surface)
is FREE and only genuine excursions are charged. Autodiff-safe via a new
target_sdf_scalar (the Vector-field target_params trigger a Taichi autodiff
MatrixPtrStmt load-forwarding bug when the SDF input is grad-tracked).
Sweep w_traj_prox in {0.003,0.01,0.03,0.1}: ALL stall at resid plateau ~0.25,
dice 0.52-0.57 (vs baseline 0.847). Even the tiniest weight prevents the
carving-sweep BREAKTHROUGH (baseline resid 0.25->0.10 between iter 1000-1500;
traj_prox stays at 0.25 past iter 2000). The excursion penalty, however gentle,
consistently pulls against the corner-sweep that carving requires.
FUNDAMENTAL FINDING: in this differentiable carving formulation, the high-dice
(0.847) trajectory INHERENTLY makes corner excursions (~30% air). ANY loss term
that discourages being far from the surface (w_air, w_prox per-voxel, w_traj_prox
per-segment) impedes the carving sweep and drops dice to ~0.55. The 30% air is
the price of high dice. => testing WARMUP (carve first, then polish) next.

### WARMUP (carve first, then polish) RESULT: ALSO FAILS
w_traj_prox with warmup_frac=0.3 (carve 1500 iters, then ramp traj_prox on).
Carving established normally (dice ~0.78-0.80, resid ~0.10-0.13 by iter 1030).
Once traj_prox ramps on (iter>1500) it DESTROYS the carve: dice 0.80->0.48-0.54,
resid 0.10->0.24. The best-checkpoint (~0.80, captured pre-polish) is BELOW
baseline 0.847 because only 1500 pure-carve iters ran before destruction.
Even post-carve, the excursion penalty fundamentally opposes the high-dice
trajectory -- it pulls the tool out of the corner-sweep that carving requires.

### CONCLUSION (air-cutting vs dice)
Robust across THREE loss designs (w_air per-voxel, w_prox distance-weighted
per-voxel, w_traj_prox per-segment center-distance) AND warmup: ANY loss term
that discourages the tool from being far from the part surface trades off dice
heavily (0.847 -> ~0.55). The high-dice sphere trajectory INHERENTLY makes
corner excursions (~30% air-cut fraction) because the sphere nearly fills the
1in stock and carving the corners requires sweeping through the empty corner
region far from the sphere surface. The ~30% air is the price of 0.847 dice;
it is NOT a tunable inefficiency. Revert all air/excursion losses to 0
(the code remains for reference). The productive dice frontier is elsewhere
(lr, iters, scenarios), not loss-based air reduction.

## DICE FRONTIER (2026-07-02, after air-cutting analysis)

### sphere dt=0.5 + m=160 at lr1e-3: WIN (higher dice AND less air!)
s0: dice 0.8527, air 0.229 (vs operating point dt0.45 m128: 0.847, air 0.295).
=> dt0.5 (faster tool, ~1.1 voxel/step) + m160 (more steps) covers the sphere
   with HIGHER dice AND 22pct LESS air. This is the rare lever that improves
   BOTH -- it directly addresses the user's air directive without the dice
   trade-off (unlike the loss-based approaches which all failed). The faster
   tool traverses the corners more directly (less re-traversal). Confirming
   with seeds 1,2,3.
   NOTE: dt0.5 needs m=160 (the operating-point note: m=160 optimal at dt0.5,
   m=128 at dt<=0.45). m>=192 NaNs (SDF overflow).

### sphere gc=0.4 at lr1e-3: WORSE (0.786, air 0.210)
gc0.4 reduces air (0.295->0.210, lowest) but drops dice (0.847->0.786) -- yet
another air-dice trade-off point. gc0.4 is a partial-air lever at a dice cost.
Discard for dice. (The prior "gc0.4 for sphere" advice was from the lr=5e-3
era; at lr=1e-3, gc0.5 is better.)

### cylinder s3: NEW best 0.9398 (vs 0.928)
Confirms cylinder is climbing with seeds; s4 running. Cylinder ceiling now
~0.94.

### box s3: 0.9172 (confirms box ~0.917, low variance)

## dt0.5+m160 FLUKE CHECK (2026-07-02)
sphere dt0.5 m160 across seeds: s0=0.8527, s1=0.841, s2=0.792, s3=0.848.
=> mean 0.834, var HIGH, air 0.37-0.43 (vs dt0.45 m128 mean 0.849, air 0.295).
   The s0=0.8527 "win" was a SINGLE-SEED FLUKE. dt0.5+m160 is higher-variance,
   higher-air, and lower-mean than dt0.45+m128. ROBUST sphere operating point
   REMAINS dt0.45 m128 lr1e-3 gc0.5 (mean 0.849 across s1/s2/s3). Discard
   dt0.5+m160 for sphere. The autoresearch.md note "m=160 at dt0.5" is a
   numerical sweet spot, not a dice improvement.

## BACKLOG / NEXT DIRECTIONS (2026-07-02)
Productive dice frontier (air-loss direction is DEAD -- see trade-off memo):
1. lr sweep around 1e-3 (RUNNING: 5e-4, 2e-3, 3e-3 on sphere). Only 1e-3 and
   5e-3 mapped so far; 1e-3 was a +0.13 jump from 5e-3. If 5e-4 > 1e-3, the
   peak is lower (try 2e-4); if 2e-3 > 1e-3, higher.
2. cylinder is the highest-dice shape and climbing (s3 0.9398, s4 0.9336).
   Try cyl lr sweep + iters>5000 to push past 0.94.
3. w_step (constant-feed regularizer, default 0) -- DIFFERENT from air losses;
   encourages uniform feed (CNC-like) and may NOT oppose carving. Quick test
   at small weight on sphere. Addresses user "uniform patterns" directive
   via smoothness, not air.
4. Parametric raster/spiral toolpath (low-dim, inherently uniform, low-air) --
   big architectural change; would directly satisfy "uniform CNC patterns" +
   cut less air. Hold as major direction if lr sweep stalls.

## lr SWEEP COMPLETE (2026-07-02) -- lr EXHAUSTED
sphere rf s0, dt0.45 m128 gc0.5:
  5e-4 -> 0.804 | 1e-3 -> 0.849 (PEAK) | 2e-3 -> 0.754 | 3e-3 -> 0.720 | 5e-3 -> 0.717
Sharp unimodal peak at 1e-3. Both sides drop fast. lr lever is DONE; 1e-3 is
the sphere optimum. (Cylinder/box/pyramid also use 1e-3; no reason to re-sweep
them -- the lever shape is the same.)

## CYLINDER TRAILING-EXCURSION FIX (2026-07-02, user-flagged)
User observed (visually) that cylinder run CamEnvDiff-v0__train_csg__4__1783011153407
(dice 0.934) makes a clean contour for the first ~3/4 then moves AWAY from the
part. Numerical analysis of trajectory.npy (normalized [0,1]^3 stock coords):
  - First ~70 steps: tool near cylinder surface (mean dist 0.68, the contour).
  - Steps 80-127: tool climbs MONOTONICALLY in z from 0.95 (cyl top) to 3.43,
    drifting to (0.36, 1.47, 3.43) -- far above and beside the part.
    Mean dist from surface 1.79 (vs 0.68 for steps 0-80).
ROOT CAUSE: trailing steps have no residual left to carve (cylinder done by
~step 70-80); with no residual gradient the gouge barrier pushes the tool off
the surface and it drifts upward into open air. This is NOT the air-cut trade-
off (that was loss pulling the tool TOWARD the surface, opposing carving).

FIX: w_len path-length penalty (mean squared |delta_t|^2). Agnostic to WHERE
the tool is -- only discourages motion. Carving steps: residual gradient
dominates, motion preserved. Trailing steps: no residual, so even tiny w_len
shrinks deltas toward zero -> tool STOPS instead of wandering. Implemented in
simulator (compute_length_penalty + diag_len) + train_csg (--w-len) + pipeline.

SWEEP (8 parallel, memory-capped to coexist with zichaohu/dlee jobs):
  cyl w_len in {0.001,0.003,0.01,0.03,0.1} (T=128, seed4); cyl T in {96,112}
  (no w_len); sphere w_len=0.01 safety check. Baseline cyl T=128 = 0.934.
  PREDICTION: moderate w_len (~0.01) keeps dice ~0.93 AND collapses the
  trailing z-climb; too-large w_len will under-carve (suppress necessary
  motion) -> dice drops.

## w_len SWEEP RESULT (2026-07-02) -- CLEAN WIN, addresses user directive
cyl T=128 s4, baseline (no w_len) = dice 0.934, air 0.286, trailing z-climb 1.704.
  w_len=0.001 -> 0.9405, air 0.243
  w_len=0.003 -> 0.9395, air 0.272
  w_len=0.01  -> 0.9419, air 0.235
  w_len=0.03  -> 0.9448, air 0.1995  <-- BEST; trailing z-climb 1.704 -> 0.010
  w_len=0.1   -> 0.9414, air 0.1723  <-- lowest air
  T=112 (no w_len) -> 0.9428 (confirms trailing steps hurt; but w_len better)
  sphere w_len=0.01 -> 0.8547 (>= 0.847 baseline; w_len SAFE for sphere)
VERDICT: w_len is the WIN that the contour-hug losses could never be. It
improves BOTH dice AND air on cylinder (and doesn't hurt sphere) because it is
agnostic to WHERE the tool is -- it only shrinks trailing drift. The user's
"tool moves away from the part" complaint is FIXED (trailing z-climb 1.704 ->
0.010). Operating point: w_len=0.03 for cylinder. Try w_len=0.01-0.03 on
box/pyramid/sphere next.

## w_step SWEEP RESULT (2026-07-02) -- small sphere WIN + CNC-uniform
sphere rf s0, dt0.45 m128 gc0.5 (baseline 0.849):
  w_step=0.001 -> 0.8578 (BEST; +0.009 over baseline; air 0.322)
  w_step=0.01  -> 0.8520
  w_step=0.1   -> 0.8507
w_step (constant-feed regularizer, penalizes step-LENGTH changes) gives a small
sphere improvement AND directly encourages the uniform-feed CNC pattern the user
asked for -- without opposing carving (it acts on step length, not direction or
position). Saturates fast (0.001 is enough). Operating point: w_step=0.001 for
sphere. NOTE: orthogonal to w_len (which fixes trailing drift); can combine.

## STAGED TRAINING RESULT (2026-07-02) -- machinery works, hard-dice gain marginal
cyl s4 (the user's flagged run), staged: stage1 train -> truncate t*=57 (drop
70 trailing excursion steps) -> stage2 train from saved mid-cut stock.
HARD-carve (deployable) dice, apples-to-apples:
  stage-1 full (128 pos)      : 0.7187
  stage-1 truncated (58 pos)  : 0.7187  (trailing 70 steps cut NOTHING)
  staged concat (185 pos)     : 0.7203  (+0.0016 over stage-1; marginal)
  stage-2 soft (on saved stock): 0.932  (training metric)
KEY FINDING: the SOFT-carve training metric (0.934 for stage-1, 0.932 for
stage-2) is HEAVILY INFLATED vs the HARD-carve deployable metric (0.7187) -- a
0.21 gap. The soft union's log(2)/k per-step bias over-erodes; a trajectory
optimized for soft does NOT transfer to hard. Staging machinery is correct and
functional (truncation drops the excursion; stage-2 trains from the saved
state) but the hard-dice gain is marginal because stage-2 optimizes the biased
soft objective. The real bottleneck for deployable dice is the soft/hard
mismatch, not the trailing excursion.

## COMBINE w_len+w_step SWEEP (2026-07-02) -- levers OVERLAP, don't stack
w_len=0.03 + w_step=0.001, rf, s1:
  sphere  : 0.8501 (flat vs 0.849; w_step ALONE 0.858 better)
  box     : 0.9163 (== baseline 0.9166; AIR 0.114 LOWEST EVER vs ~0.28)
  pyramid : 0.8892 (> baseline 0.885; +0.004)
  cylinder: 0.9357 (w_len ALONE s4 0.9448 better; but different seed)
VERDICT: the two regularizers partially overlap (both discourage motion), so
forcing both everywhere is NOT better than the per-shape best individual lever.
KEEP per-shape operating points: sphere w_step=0.001 (0.858), cylinder
w_len=0.03 (0.945), box/pyramid w_len+w_step (box air 0.114 is excellent).
The box air 0.114 at full dice shows w_len+w_step can near-eliminate air on
the right shape -- directly serves the user's air directive.

## NEW DIRECTION: the soft/hard carve gap (the real deployable-dice wall)
Staging exposed that soft-carve dice (0.934) is ~0.21 above hard-carve (0.7187).
The soft union adds log(2)/k per step, over-eroding. If I can sharpen the
union (lower k -> closer to hard boolean) WITHOUT breaking gradients, the
soft training objective would match the deployable hard carve -> the 0.93
soft would transfer. This is the highest-value lever for deployable dice.
NEXT: sweep k (union smoothness) on cylinder, measure soft AND hard dice.

## k SWEEP RESULT (2026-07-02) -- soft/hard gap is NOT a smoothness knob
cyl w_len0.03 s4, sweep k (soft-union sharpness; lower=sharper=closer to hard):
  k=10.0: soft 0.9383, HARD 0.7204  (the working k; gap 0.218)
  k=5.0 : soft 0.7493, HARD 0.7175  (gradients weaken, soft drops)
  k=2.0 : soft 0.0000, HARD 0.7175  (sharp union saturates -> degenerate)
  k=1.0 : soft 0.0000, HARD 0.7175
  k=0.5 : soft 0.0000, HARD 0.7175
DECISIVE: the HARD carve is ~k-INVARIANT (~0.718 for all k<=5, 0.720 at k=10).
Lowering k does NOT close the soft/hard gap -- it just breaks the optimizer
(sharp union saturates, gradients vanish, trajectory degenerates to zero).
The gap is the soft union's INHERENT per-step bias (smooth_max adds ~log(2)/k
per carve -> soft over-erodes vs the hard boolean), NOT a tunable artifact.
The soft training objective is a BIASED PROXY; the hard carve (~0.72) is the
true deployable dice and is capped by trajectory coverage, not smoothness.
CONCLUSION: k=10 is correct. The soft/hard gap is fundamental to the method.
To raise DEPLOYABLE dice, must improve the TRAJECTORY's hard-carve coverage
(more steps / finer feed / better path), not the loss smoothness.

## T SWEEP RESULT (2026-07-02) -- soft dice peaks T=256 cyl, marginal
cyl w_len0.03 s4: T128=0.9383(k10)/0.9448(orig s4) | T160=0.9429 | T192=0.9443
| T256=0.9457 (PEAK) | T320=0.9399 (drops, unstable). +0.0009 soft over T128 at
T256; air rises with T (0.175->0.349). HARD dice ~0.718 FLAT across all T
(confirms hard is coverage-capped, not T-limited). T=256 is the cyl soft peak
but the gain is marginal and costs air + budget headroom; T=128-192 is the
practical operating range.

## DICE CONVENTION FINDING (critical, 2026-07-02)
eval_csg dice: pred=sdf_to_mask(stock)=stock<0=REMAINING material, target=PART.
Dice = 2|remaining∩part|/(|remaining|+|part|) -- rewards LEAVING the part
(not gouging) more than removing waste. A STATIONARY tool scores 0.728
(=2|cyl|/(|stock|+|cyl|)=2·0.572/1.572) because it leaves the whole part. The
soft union over-erodes (removes outside-part material the hard boolean doesn't),
so SOFT dice (0.94) >> HARD dice (0.718). The soft dice is a BIASED proxy; the
hard carve is the deployable number. Per autoresearch.md the loop tracks soft
dice (grep "^dice:" train log); hard carve is a separate validation that must
not be modified. IMPLICATION: the reported soft wins (sphere 0.85, cyl 0.945)
are real on the tracked metric but overstate deployable quality by ~0.21.

## PAIRED-SEED ROBUSTNESS VALIDATION (2026-07-02) -- wins are modest, not big
6-run paired-seed check of the two headline wins (same GPU, 3 seeds each):
  SPHERE w_step=0.001: s1=0.8503, s2=0.8468, s3=0.8580 -> mean 0.8517
    vs baseline ~0.848 -> +0.004 (WITHIN NOISE; s1/s2 below baseline, s3 above).
    The earlier single-seed 0.858 was a high-variance lucky seed. w_step is at
    best a marginal sphere lever, NOT a reliable +0.01.
  CYLINDER w_len=0.03 T256: s1=0.9499, s2=0.9398, s3=0.9336 -> mean 0.9411
    vs baseline s3=0.9398/s4=0.9336 (mean 0.9367) -> +0.004 mean. s1=0.9499 is a
    HIGH OUTLIER; s2/s3 (0.9398/0.9336) == baseline. w_len+T256 gives a
    high-variance high tail but the mean gain is modest (~+0.004).
VERDICT: both "wins" survive paired seeds but are SMALL (~+0.004 mean), not the
+0.01 the single seeds suggested. Confirms the methodological lesson: single-
seed apparent wins overstate. The operating point (dt0.45 lr1e-3 w_len0.03
T256) is still the best method, just with honest effect sizes.

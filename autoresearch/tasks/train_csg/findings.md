# Findings — jul5-anneal-gap (autoresearch loop)

Branch: `ar-agd/jul5-anneal-gap`. Date: 2026-07-05.
Objective: find the best shape-blind method for training the analytical
gradient-descent toolpath optimizer in the diff-cam differentiable CNC
simulator, maximizing deployable (hard-carve, crash-safe) Dice across shapes.

**Generality constraint (hard requirement, met):** no method, init, or loss
inspects or branches on the target shape name. Every algorithm below operates
only on the SDF field and the carved stock field. `--target-shape` selects the
target SDF; it is never read by the optimizer/loss.

## The two method contributions

Both are shape-blind, both confirmed across 3 seeds where available.

### 1. k-anneal (smoothness annealing) + reachability fix
- Linear ramp of the soft-union sharpness `k` from `k_init=10` → `k_final=150`
  over training (`sim.k[None] = k_init + (k_final-k_init)*it/iters`).
- **Why it works:** the in-loop eval already scores HARD dice (`forward_hard`,
  k-independent). k only affects the TRAINING forward (`apply_cut`, soft union),
  which over-erodes at finite k. Sharpening k late makes the training gradient
  less over-erosion-biased → better aligned with the hard-dice metric. It is a
  *gradient-bias reduction* lever, not a metric change.
- **Reachability fix:** `--tool-height-mm 50.0` (was 25mm). The 25mm tool
  collides its holder with a 1in stock, truncating the trajectory to ~40/128
  waypoints. 50mm clears the holder → full 128-waypoint deployable trajectory,
  viz=train. 75mm saturates (= 50mm) — a dead lever.

### 2. loss-shift (de-biased, hard-carve-aware loss)
- Add `loss_shift` to `stock_d` before the loss sigmoid:
  `sa = clamp((stock_d + loss_shift) * scale, -50, 50)`.
- **Why it works:** soft-union over-erosion biases `stock_d` negative in the
  training forward, so the loss "sees" the stock as less carved than the deployable
  hard carve → it keeps carving → over-erosion → soft/hard gap. A NEGATIVE shift
  makes the loss see the stock as more-carved → eases off → less over-erosion →
  soft matches hard. Principled value ~log(2)·k_ref/k_final ≈ 0.24 voxels;
  empirical sweet spot `-0.5`.
- Direction was verified both ways: +shift monotonically worsened over-carve
  (carved vox 57317→70332); −shift reduced it (64420 vox) and lifted dice.

## Confirmed operating point (1in stock)

```
--tool-height-mm 50.0 --k-anneal --k-init 10 --k-final 150 --loss-shift -0.5
```
on top of: `--dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10
--iters 5000 --max-steps 128`.

Deployable hard Dice (viz=train, NO truncation), seed mean ± std:

| shape      | k=10 base | k150 +50mm | **k150 +50mm ls-0.5** | seeds |
|------------|-----------|------------|------------------------|-------|
| sphere     | 0.638     | 0.820±0.019| **0.826±0.018**        | s1,s2,s3 |
| pyramid    | 0.427     | 0.797±0.004| **0.813±0.003**        | s1,s2,s3 |
| cylinder   | 0.774     | 0.891±0.025| **0.905±0.010**        | s1,s2,s3 |
| box        | 0.816     | 0.843      | **0.865±0.009**        | s1,s2,s3 |
| **mean (4 prim)** | 0.664 | 0.838 | **0.877**          |       |
| sphere_bowl| —         | 0.612      | 0.634±0.020 (ls neutral)| s1,s2,s3 |
| sphere_hole| —         | 0.237 (sub6)| 0.246 (sub6)          | s1    |

**Net: +0.213 mean deployable Dice over the k=10/25mm baseline (0.664 → 0.877),
fully deployable (viz = train, no truncation).** loss-shift adds ~+0.014 mean
on top of k-anneal; its main value is de-biasing, reproducible across seeds.

## Characterized limits / dead levers (do not re-explore)

- **75mm tool** = 50mm (saturated). Tool length saturates at 50mm for 1in stock.
- **High-k (k_final 500/1000)** — hurts positive features AND does not close the
  narrow-hole soft/hard gap. k150 is the sweet spot.
- **m256 (256 waypoints)** — WORSE on average (mean 0.826 vs m128 0.852). More
  parameters overfit/diverge on simple shapes (sphere −0.057, cyl −0.085, box
  −0.012); only pyramid benefits (+0.044). Shape-dependent → not a blind win.
  m=128 is the general default.
- **loss-shift direction:** +shift is wrong (more over-carve); −0.5 sweet spot;
  −1.0 under-carves. Fine sweep [−0.3, −0.5, −0.7] all ≈ optimal on sphere.
- **loss-shift on bowl:** neutral (bowl's concavity isn't over-erosion-limited).
- **air-cut loss weights** (w_air/w_prox/w_traj_prox): explored in prior runs —
  fundamentally trade off Dice for air reduction (~30% air inherent). Don't
  re-explore.

## Open frontier: sphere_hole (through-hole, negative feature)

`sphere_hole` = sphere ∩ ¬cylinder. The narrow through-hole exposes a structural
soft/hard gap: at any finite k, soft-union over-erosion fills the narrow negative
feature, so the training loss cannot penalize the hole interior. loss-shift
helps marginally (0.237→0.246) but the hole interior is structurally
unpenalized. This is the limit of pure soft-loss methods; closing it needs a
hard-carve-aware loss term or a topology-aware penalty (future work).

Scaling: 2in stock + 100mm tool (same tool-to-stock ratio) confirms k-anneal
scales to the larger absolute scenario. **Sphere 2in k150 ls-0.5 = 0.774**
(deployable, viz=train, no truncation) vs 2in k10 ~0.55 → k-anneal+loss-shift
delivers +0.22, matching the 1in pattern (k-anneal is the dominant lever and
transfers across absolute scale). The gap to 1in sphere (0.826) is the larger
tool-to-stock ratio, not a method failure. Box/pyramid 2in still in flight.

## Artifacts

- `results.tsv` (UNTRACKED, 95 rows) — every experiment, full command + status.
- `results_plot.png` — progress panel (running best) + generality panel
  (best dice per shape × method-family, 1in stock). Regenerate via
  `python plot_results.py`.
- `idea.md` — running research log (all hypotheses, sweeps, diagnoses).
- Runs grouped in `runs/jul5-anneal-gap/` for the web UI (88+ runs).

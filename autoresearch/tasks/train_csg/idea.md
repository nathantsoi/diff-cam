# jul6-step-detail — working log

Branch: `ar-agd/jul6-step-detail` (from `ar-agd/jul6-spline-sweep` @ be65551 —
inherits the one-shot spline-swept-volume method, its tests, and
`future_work.md`). Worktree: `.claude/worktrees/step-detail`.
**Every run needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`** (WSL2 CUDA).

## Goal (user-directed)

**Maximize hard-carve dice on real STEP-file targets** (`--target-shape grid
--target-sdf-path <npz>` from `utils/step_to_sdf.py`), adapting the spline
sweep method to detailed parts as needed, guided by `future_work.md` (kept
from the previous campaign as the exploration basis). All method rules still
apply: shape-agnostic optimizer/init/losses (geometric representation only —
SDF grids, bboxes, reachability masks derived from geometry), 15-min budget,
eval untouched.

## Targets (regenerated NPZs, padding 0 = stock is the exact part bbox)

| npz | part | phys mm | voxel | grid | 3-axis ceiling |
|---|---|---|---|---|---|
| titan_hi | Titan-M8 nameplate (1 solid) | 138×49×19 | 0.5 | 277×99×38 (1.0M) | **0.965** |
| rrph_hi | RoundedRect+Pin&Hole, **solid 0 only** | 25×51×13 | 0.3 | 85×169×42 (0.6M) | **0.970** |
| extrusion_hi | Extrusion (lies flat, z=20) | 20×100×20 | 0.5 | 40×200×40 (0.3M) | **0.648** |
| bowl_hi | bowl + 3 feet, **y↔z swapped** (model y-up) | 260×260×76 | 1.5 | 173×173×51 (1.5M) | **0.342** |

Why regenerated: pre-existing `RoundedRectangleHighRes.npz` included all 11
solids — 1–10 are giant construction geometry that exploded the stock to
2.2 m. Solid 0 is the real 1×2×0.5 in part. Padding 0 removes the symmetric
under-part stock slab; part sits flush at stock bottom like real fixturing.
Conversion commands (main-repo venv has OCP bindings; worktree venv doesn't):
`uv run python utils/step_to_sdf.py <step> -o utils/NPZs/<name>.npz
--voxel-size-mm <v> --padding 0 [--solid-indices 0]` + y↔z transpose for bowl.

**Ceilings** computed with a shape-agnostic exact reachability mask (part
height field max-filtered by the tool disc; a waste voxel is removable iff
some tool position covers it without the cylinder clipping part above):
- titan/rrph ~0.97: clean targets, most waste reachable.
- extrusion 0.648 (saturates ~0.69 even with r→0 tool): profile cavities open
  SIDEWAYS (part lies flat) — orientation shadow, not tool width.
- bowl 0.342: curved underside is one big overhang shadow (real shops flip
  the part). Stress test only; interpret dice vs its ceiling.
The same mask is a candidate METHOD lever: gate w_broad attraction/residual
loss off unreachable waste so gradient isn't wasted on impossible voxels
(future_work "exact vertical-accessibility mask", DeepMill discussion).

## Key structural constraint: path-length budget

Executable path ≤ T × feed·dt = T × 1.905 mm (dt 0.45, feed 10 ipm).
T=256 → 488 mm. One boustrophedon layer over titan's 138×49 top at ~4.7 mm
stepover ≈ 1.7 m. Detailed parts are path-length starved at T=256 → first
adaptation axis is scaling T (and K ∝ T) with part size; sweep cost is
O(T·N³)/iter so T=1024 on ~1M voxels ≈ 100 ms/iter ≈ 9k iters in budget.
dt is the cheap alternative knob (longer step cap) but coarsens spline
sampling — chord sagitta risk on tight curves; test empirically.

## Plan

1. Infra (done): run_pipeline grid+target-sdf-path forwarding; NPZs; ceilings.
2. Baselines: sweep UNCHANGED (T256 K40 lr1e-3 w_broad0.1) on titan + rrph;
   delta reference on titan. Expect path starvation.
3. Scale T/K (512/1024/2048, K∝T/6), dt probe; derive auto-scaling rule.
4. future_work levers by payoff: reachability-gated attraction, multi-spline
   safe-z retracts (pin+hole plunge), multi-start diverse inits, slope-aware
   |N_z| weighting (bowl), curriculum loss, path-side residual attraction.
5. extrusion + bowl with ceiling-aware expectations; ≥3 seeds on headline
   config; plot + findings.

## Log

- (setup) Worktree + branch; results.tsv truncated (untracked); prior
  findings.md removed (preserved on spline-sweep branch); runs/jul6-step-detail/.
- run_pipeline: added `grid` target choice, `--target-sdf-path` forwarding,
  `--stock-size-in` now optional (grid targets take the box from the NPZ).
- STEP solids audited: titan 1 solid 138×49×19; rrph solid0 25×51×13 (+10
  junk up to 1.8 m); extrusion 20×100×20 y-long lying flat; bowl 4 solids
  (bowl + 3 feet) y-up → swapped to z-up.
- NPZs generated (seconds each); bowl wall is thin (SDF min −1.75 mm at
  1.5 mm voxels — 2-voxel shell, precision-sensitive).
- Reachability ceilings (table above); extrusion tool-radius sweep shows
  saturation at ~0.69 → orientation shadow, smaller tools don't help.

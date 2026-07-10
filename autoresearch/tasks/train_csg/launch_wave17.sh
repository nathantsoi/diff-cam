#!/bin/bash
# Wave 17: best_on_hard + w_tool_gouge (soft-union-INDEPENDENT gouge barrier).
# Wave-16 finding (see memory best-on-hard-gouge-tradeoff.md): --best-on-hard
# captures final-iter hard dice (gap≈0: pyr 0.674, sph 0.777) BUT deploys a
# heavily-gouging checkpoint (pyr gouge 509, sph 572, bowl 921) because
# composite_score penalizes air/time/break but NOT gouge. The stock-based
# w_gouge is satisfied by soft-union over-erosion while the HARD carve still
# gouges, so it does NOT transfer. w_tool_gouge charges the tool CENTER
# directly for penetrating target (+r_tool): relu(r_tool - target_sdf)^2 --
# zero tangent-or-outside, grows with penetration -- so it constrains the
# trajectory GEOMETRY and transfers to the hard carve.
#
# GOAL: high hard dice (≈ best_on_hard levels) WITH gouge back near soft
# levels (pyr ~60, sph ~0). Scout wave: 1 repeat, sweep w_tool_gouge scale
# on the 3 gouge-blowup shapes (pyr/sph/bowl) + 1 hole control. Wave 18 will
# confirm the winning scale with >=3 repeats (variance rule, see
# csg-run-variance-floor.md).
# 8 runs, seed=1, 5000 iters canonical K70, all --best-on-hard.
# Usage: bash launch_wave17.sh [first_gpu]
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10 --runs-subdir jul8-multidepth"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
SPH="--target-shape sphere --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <iters> <args...>
  local gpu="$1"; local name="$2"; local iters="$3"; shift 3
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters "$iters" --max-steps 128 $COMMON --best-on-hard "$@" > "$log" 2>&1 &
  echo "[wave17] GPU$gpu -> $name (iters=$iters)"
}

# w_tool_gouge scale sweep. All runs are best_on_hard + canonical K70.
# pyr spans the low/mid/high end {2,8,32}; sph/bowl get {8,32}; hole control at 8.
launch 0 w17_pyr_tg2   5000 $PYR  $MD5 $K70 --w-tool-gouge 2.0
launch 1 w17_pyr_tg8   5000 $PYR  $MD5 $K70 --w-tool-gouge 8.0
launch 2 w17_pyr_tg32  5000 $PYR  $MD5 $K70 --w-tool-gouge 32.0
launch 3 w17_sph_tg8   5000 $SPH  $MD5 $K70 --w-tool-gouge 8.0
launch 4 w17_sph_tg32  5000 $SPH  $MD5 $K70 --w-tool-gouge 32.0
launch 5 w17_bowl_tg8  5000 $BOWL $MD5 $K70 --w-gouge 6.0 --w-tool-gouge 8.0
launch 6 w17_bowl_tg32 5000 $BOWL $MD5 $K70 --w-gouge 6.0 --w-tool-gouge 32.0
launch 7 w17_hole_tg8  5000 $HOLE $HOLEBASE $K70 --w-tool-gouge 8.0

echo "[wave17] all 8 launched. Logs in $D/run_w17_*.log"
echo "[wave17] scout wave (1 repeat). Confirm winning w_tool_gouge in wave 18 with >=3 repeats."

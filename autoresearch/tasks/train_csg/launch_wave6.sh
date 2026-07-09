#!/bin/bash
# Wave 6: 8-wide. (a) Complete the matched generality table with random+wr5
# baselines on sphere+cylinder (wave-4 random baselines were wr1, not wr5).
# (b) Attack the sphere_hole collapse where BOTH methods sit near 0 -- the
# through-hole is a concave void multidepth's exterior-ring init can't open
# (random already edges multidepth there 0.096 vs 0.048). Levers: k-anneal/high-k
# (shrinks soft-union bias so the soft loss tracks HARD coverage on the concave
# hole), higher w_residual, and w_traj_prox (tool-center contour-hug to find the
# hole wall). (c) Probe the now-CORRECTED air-time loss lever (w_air_time 10x) on
# the working sphere config -- can we cut air without killing dice now that the
# metric is honestly wired?
# Confirmed losers (excluded): loss_shift, w_gouge>4, lr>5e-3.
# Usage: bash launch_wave6.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
MD="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
RAN="--init-mode random --w-residual 5.0"
SPHERE="--target-shape sphere --target-radius-mm 11.43"
CYL="--target-shape cylinder --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave6] GPU$gpu -> $name"
}

# (a) Matched random+wr5 baselines (complete the 6-shape table).
launch 0 w6_sph_rand5   $SPHERE $RAN
launch 1 w6_cyl_rand5   $CYL    $RAN

# (b) sphere_hole attack. Random is the better init here (it can stumble into the
# interior hole); give it the k-sharpening + residual levers, plus an md
# contour-hug variant. k-anneal 10->60 = continuation (smooth explore early,
# hard-track late).
launch 2 w6_hole_rand_kann      $HOLE $RAN --k-anneal --k-init 10 --k-final 60
launch 3 w6_hole_rand_k60       $HOLE $RAN --k-init 60
launch 4 w6_hole_rand_wr10      $HOLE $RAN --w-residual 10.0
launch 5 w6_hole_rand_wr10_kann $HOLE $RAN --w-residual 10.0 --k-anneal --k-init 10 --k-final 60
launch 6 w6_hole_md_tprox       $HOLE $MD  --w-traj-prox 1.0 --w-traj-prox-warmup-frac 0.5

# (c) Air-lever probe on the working sphere config: raise w_air_time 10x.
launch 7 w6_sph_md_air10        $SPHERE $MD --w-air-time 1e-2

echo "[wave6] all 8 launched. Logs in $D/run_w6_*.log"

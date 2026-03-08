#!/bin/bash
#----------------------------------------------------
# SLURM job script for diff-cam PPO training
#   TACC Lonestar6 — GPU A100 node
#
# Usage:
#   cd $WORK/diff-cam
#   sbatch scripts/train.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm-<jobid>.out
#----------------------------------------------------

#SBATCH -J diffcam-ppo         # Job name
#SBATCH -o diffcam-ppo.o%j     # stdout  (%j = jobid)
#SBATCH -e diffcam-ppo.e%j     # stderr
#SBATCH -p gpu-a100             # Queue — A100 GPU nodes
#SBATCH -N 1                    # Total nodes
#SBATCH -n 1                    # Total MPI tasks (1 for single-process training)
#SBATCH -t 02:00:00             # Wall-clock time (hh:mm:ss)
#SBATCH --mail-type=all         # Email at begin/end/fail
##SBATCH --mail-user=you@utexas.edu   # ← uncomment and set your email
##SBATCH -A myproject            # ← uncomment and set your allocation

# ── Modules ──────────────────────────────────────────
module load gcc/13.2.0 cuda/12.8

# ── Paths ────────────────────────────────────────────
DIFF_CAM_DIR="${DIFF_CAM_DIR:-$SCRATCH/diff-cam}"
cd "$DIFF_CAM_DIR"

# Activate the venv (symlinked from $SCRATCH)
source .venv/bin/activate

# ── Run Training ─────────────────────────────────────
echo "=== Starting PPO training ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPUs   : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Python : $(python --version)"
echo "CWD    : $(pwd)"
echo ""

python -u algorithms/ppo.py \
    --cuda \
    --total-timesteps 2000000 \
    --num-envs 3 \
    --num-steps 256

echo ""
echo "=== Training complete ==="
echo "Date: $(date)"

#!/bin/bash
#----------------------------------------------------
# SLURM job script for diff-cam PufferLib training
#   TACC Lonestar6 — GPU A100 node
#
# Usage:
#   cd $WORK/diff-cam
#   sbatch scripts/train_puffer.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f diffcam-puffer.o<jobid>
#----------------------------------------------------

#SBATCH -J diffcam-puffer      # Job name
#SBATCH -o diffcam-puffer.o%j  # stdout  (%j = jobid)
#SBATCH -e diffcam-puffer.e%j  # stderr
#SBATCH -p gpu-a100-small      # Queue — A100 GPU nodes
#SBATCH -N 1                   # Total nodes
#SBATCH -n 1                   # Total MPI tasks (1 for single-process training)
#SBATCH -t 24:00:00            # Wall-clock time (hh:mm:ss)
#SBATCH --mail-type=all        # Email at begin/end/fail
#SBATCH --mail-user=aditya.angajala@utexas.edu   # ← uncomment and set your email
#SBATCH -A IRI25030            # ← uncomment and set your allocation

# ── Modules ──────────────────────────────────────────
module load gcc/13.2.0 cuda/12.8

# ── Paths ────────────────────────────────────────────
DIFF_CAM_DIR="${DIFF_CAM_DIR:-$SCRATCH/diff-cam}"
cd "$DIFF_CAM_DIR"

# Activate the venv (symlinked from $SCRATCH)
source .venv/bin/activate

# ── Run Training ─────────────────────────────────────
echo "=== Starting PufferLib training ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPUs   : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Python : $(python --version)"
echo "CWD    : $(pwd)"
echo ""

python -u algorithms/train_pufferl.py \
    --num-envs 24 \
    --resolution 64 \
    --total-timesteps 50000000

echo ""
echo "=== Training complete ==="
echo "Date: $(date)"

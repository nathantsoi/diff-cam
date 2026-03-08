#!/bin/bash
#SBATCH -J rl-training
#SBATCH -o output/train_%j.txt
#SBATCH -e output/train_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-small
#SBATCH -t 04:00:00
#SBATCH -A IRI25030

module load tacc-apptainer

singularity exec --nv $SCRATCH/diff-cam.sif \
    python3 /app/algorithms/ppo.py \
    --total_timesteps 10000000 \
    --num_envs 16 \
    --num_steps 2048 \
    --max_steps 1024
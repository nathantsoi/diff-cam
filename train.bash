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

singularity exec --nv \
    --bind $SCRATCH/diff-cam/runs:/app/runs \
    --bind $SCRATCH/diff-cam/simulator:/opt/conda/lib/python3.11/site-packages/simulator \
    --bind $SCRATCH/diff-cam/cam_env:/opt/conda/lib/python3.11/site-packages/cam_env \
    --bind $SCRATCH/diff-cam/algorithms:/app/algorithms \
    $SCRATCH/diff-cam.sif \
    python3 /app/algorithms/ppo.py \
        --total_timesteps 10000000 \
        --num_envs 1 \
        --resolution 32 \
        --num_steps 2048 \
        --max_steps 4096 \
        --num_minibatches 8 \
        --update_epochs 4 \
        --learning_rate 3e-4 \
        --ent_coef 0.02 \
        --gamma 0.99 \
        --gae_lambda 0.95
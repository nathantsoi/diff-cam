#!/bin/bash
# Local training for the CONTINUOUS (CSG / GradMill) PPO baseline.
# Run from the repo root. Scripts must be launched as modules (python -m ...)
# so that `cam_env` / `simulator` resolve on the import path.

uv run python -m algorithms.csg_ppo \
        --total_timesteps 10000000 \
        --num_envs 1 \
        --resolution 32 \
        --max_steps 64 \
        --num_steps 512 \
        --num_minibatches 8 \
        --update_epochs 4 \
        --learning_rate 3e-4 \
        --ent_coef 0.02 \
        --gamma 0.99 \
        --gae_lambda 0.95 \
        --save_model
        # Optional: record a greedy policy rollout video every N iterations and
        # upload it to wandb (requires --track). e.g.:
        # --record_video_freq 25 --video_fps 30

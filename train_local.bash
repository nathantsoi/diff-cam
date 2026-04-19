#!/bin/bash

uv run python3 ./algorithms/ppo.py \
        --total_timesteps 10000000 \
        --num_envs 4 \
        --resolution 32 \
        --num_steps 2048 \
        --max_steps 4096 \
        --num_minibatches 8 \
        --update_epochs 4 \
        --learning_rate 3e-4 \
        --ent_coef 0.02 \
        --gamma 0.99 \
        --gae_lambda 0.95 \
        --render_mode human
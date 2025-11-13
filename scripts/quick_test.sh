#!/bin/bash
# Quick test of RQE-MAPPO training

echo "Running quick training test..."

uv run python -m src.train \
    --env simple_spread \
    --n_agents 3 \
    --total_timesteps 10000 \
    --batch_size 512 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --tau 0.5 \
    --epsilon 0.01 \
    --exp_name test_run

echo "Training complete! Check logs/test_run for results."

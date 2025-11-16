#!/bin/bash
# Quick test of RQE-MAPPO on SUMO (10K steps, ~5 min)

echo "Quick test of RQE-MAPPO on SUMO..."
echo ""

export SUMO_HOME="/usr/share/sumo"

uv run python -m src.train_sumo \
    --net single-intersection \
    --total_timesteps 10000 \
    --batch_size 512 \
    --num_seconds 200 \
    --tau 0.5 \
    --epsilon 0.01 \
    --exp_name test_sumo_intersection

echo ""
echo "Test complete! Check logs/test_sumo_intersection for results."

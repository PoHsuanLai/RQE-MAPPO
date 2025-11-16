#!/bin/bash
# Train MAPPO baseline using Stable Baselines3 on SUMO

NETWORK=${1:-single-intersection}  # single-intersection, 2x2grid, 4x4-Lucas, etc.

echo "Training MAPPO Baseline (Stable Baselines3) on SUMO network: $NETWORK"
echo ""

export SUMO_HOME="/usr/share/sumo"

uv run python -m src.train_sb3_mappo \
    --net $NETWORK \
    --total_timesteps 500000 \
    --n_steps 2048 \
    --num_seconds 1000 \
    --delta_time 5 \
    --lr 3e-4 \
    --ent_coef 0.01 \
    --exp_name sb3_mappo_${NETWORK}

echo ""
echo "Training complete! Check logs/sb3_mappo/ for results."
echo ""
echo "To compare with RQE-MAPPO, run:"
echo "  ./scripts/train_sumo.sh $NETWORK 0.5 0.01"

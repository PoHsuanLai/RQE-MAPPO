#!/bin/bash
# Train RQE-MAPPO on SUMO environments

NETWORK=${1:-single-intersection}  # single-intersection, 2x2grid, 4x4-Lucas, etc.
TAU=${2:-0.5}                       # Risk aversion
EPSILON=${3:-0.01}                  # Bounded rationality

echo "Training RQE-MAPPO on SUMO network: $NETWORK"
echo "Risk aversion (tau): $TAU"
echo "Bounded rationality (epsilon): $EPSILON"
echo ""

export SUMO_HOME="/usr/share/sumo"

uv run python -m src.train_sumo \
    --net $NETWORK \
    --total_timesteps 500000 \
    --batch_size 2048 \
    --num_seconds 1000 \
    --tau $TAU \
    --epsilon $EPSILON \
    --exp_name sumo_${NETWORK}_tau${TAU}_eps${EPSILON}

echo ""
echo "Training complete! Check logs/sumo_${NETWORK}_tau${TAU}_eps${EPSILON} for results."

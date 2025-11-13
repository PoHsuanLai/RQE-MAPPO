#!/bin/bash
# Train RQE-MAPPO on traffic scenarios

SCENARIO=${1:-intersection}  # intersection, merge, or passing
TAU=${2:-0.5}                 # Risk aversion
EPSILON=${3:-0.01}            # Bounded rationality

echo "Training RQE-MAPPO on traffic scenario: $SCENARIO"
echo "Risk aversion (tau): $TAU"
echo "Bounded rationality (epsilon): $EPSILON"
echo ""

uv run python -m src.train_traffic \
    --scenario $SCENARIO \
    --n_vehicles 3 \
    --total_timesteps 500000 \
    --batch_size 2048 \
    --eval_interval 25000 \
    --save_interval 100000 \
    --tau $TAU \
    --epsilon $EPSILON \
    --exp_name traffic_${SCENARIO}_tau${TAU}_eps${EPSILON}

echo ""
echo "Training complete! Check logs/traffic_${SCENARIO}_tau${TAU}_eps${EPSILON} for results."

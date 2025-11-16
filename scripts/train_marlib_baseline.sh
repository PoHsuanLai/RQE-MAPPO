#!/bin/bash
# Train standard MAPPO baseline using EPyMARL on SUMO environments

NETWORK=${1:-single-intersection}  # single-intersection, 2x2grid, 4x4-Lucas, etc.

echo "Training MAPPO Baseline (EPyMARL) on SUMO network: $NETWORK"
echo ""

export SUMO_HOME="/usr/share/sumo"

cd epymarl

uv run python src/main.py \
    --config=mappo \
    --env-config=sumo \
    with \
    env_args.net=$NETWORK \
    env_args.num_seconds=1000 \
    env_args.delta_time=5 \
    t_max=500000 \
    test_interval=25000 \
    log_interval=10000

cd ..

echo ""
echo "Training complete! Check epymarl/results/ for logs."
echo ""
echo "To compare with RQE-MAPPO, run:"
echo "  ./scripts/train_sumo.sh $NETWORK 0.5 0.01"

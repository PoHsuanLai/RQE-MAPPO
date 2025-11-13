#!/bin/bash
# Quick test of RQE-MAPPO on traffic scenarios

echo "Testing RQE-MAPPO on intersection scenario..."
echo ""

uv run python -m src.train_traffic \
    --scenario intersection \
    --n_vehicles 3 \
    --total_timesteps 10000 \
    --batch_size 512 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --tau 0.5 \
    --epsilon 0.01 \
    --exp_name test_traffic_intersection

echo ""
echo "Test complete! Check logs/test_traffic_intersection for results."

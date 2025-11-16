#!/bin/bash

# Monitor CliffWalk training runs

echo "========================================================================"
echo "CliffWalk Training Monitor"
echo "========================================================================"
echo ""

# Check if processes are running
echo "=== Running Processes ==="
ps aux | grep -E "train_(true_)?rqe_cliffwalk" | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15, $16}'
echo ""

# Show latest results from TRUE RQE-PPO
echo "=== TRUE RQE-PPO (last 20 lines) ==="
if [ -f /tmp/true_rqe_cliffwalk_100k.log ]; then
    tail -20 /tmp/true_rqe_cliffwalk_100k.log | grep -E "(Trial|episode_reward|timesteps_total|completed|TERMINATED)" || tail -20 /tmp/true_rqe_cliffwalk_100k.log
else
    echo "Log file not found"
fi
echo ""

# Show latest results from RQE-PPO Approximation
echo "=== RQE-PPO Approximation (last 20 lines) ==="
if [ -f /tmp/rqe_approx_cliffwalk_100k.log ]; then
    tail -20 /tmp/rqe_approx_cliffwalk_100k.log | grep -E "(Trial|episode_reward|timesteps_total|completed|TERMINATED)" || tail -20 /tmp/rqe_approx_cliffwalk_100k.log
else
    echo "Log file not found"
fi
echo ""

echo "========================================================================"
echo "To view full logs:"
echo "  TRUE RQE-PPO:        tail -f /tmp/true_rqe_cliffwalk_100k.log"
echo "  Approximation:       tail -f /tmp/rqe_approx_cliffwalk_100k.log"
echo ""
echo "To kill training:"
echo "  pkill -f train_true_rqe_cliffwalk"
echo "  pkill -f train_rqe_cliffwalk"
echo "========================================================================"

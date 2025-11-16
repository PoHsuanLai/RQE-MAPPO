#!/bin/bash

# Training script for TRUE RQE-PPO on SUMO 4x4 grid

# Set SUMO environment
export SUMO_HOME=/usr/share/sumo

# Navigate to the baseline directory
cd /home/r13921098/RQE-MAPPO/sumo-rl/sumo_rl_baseline

# Create outputs directory
mkdir -p outputs

# Run TRUE RQE-PPO training
# Default: tau=1.0 (moderate risk aversion), epsilon=0.1 (bounded rationality)
uv run python train_true_rqe_ppo.py \
    --tau 1.0 \
    --epsilon 0.1 \
    --n_atoms 51 \
    --v_min -10.0 \
    --v_max 10.0 \
    --critic_loss_coeff 1.0 \
    --num_workers 2 \
    --num_gpus 0 \
    --train_batch_size 4000 \
    --sgd_minibatch_size 128 \
    --num_sgd_iter 10 \
    --lr 5e-5 \
    --stop_timesteps 100000 \
    --checkpoint_freq 10 \
    --num_seconds 3600

echo "Training completed! Check ray_results/ for outputs."

# Visualization Guide for RQE-MAPPO

This guide explains how to visualize your training results.

## Quick Start

After training, visualize results with:

```bash
# Basic training curves
uv run python -m src.visualize --exp_name YOUR_EXP_NAME --save_fig

# Full visualization with episode rendering
uv run python -m src.visualize --exp_name YOUR_EXP_NAME --render --n_episodes 5 --save_fig
```

## Available Visualizations

### 1. Training Curves (`training_curves.png`)

Shows 6 key metrics over training:
- **Episode Reward**: Total reward per episode (should increase)
- **Collision Rate**: Percentage of episodes with collisions (should decrease)
- **Actor Loss**: Policy gradient loss
- **Critic Loss**: Value function loss
- **Policy Entropy**: Exploration measure (should stay positive)
- **Goals Reached**: Cumulative goal completions

### 2. Evaluation Results (`evaluation_results.png`)

Shows performance over N evaluation episodes:
- Episode rewards distribution
- Collisions per episode
- Goals reached per episode

### 3. Trajectory Plots (`trajectory_ep*.png`)

For each rendered episode:
- **Left panel**: Vehicle trajectories in 2D space
  - Circles = start positions
  - Squares = end positions
  - Lines = paths taken
- **Right panel**: Rewards over time
  - Individual vehicle rewards
  - Cumulative reward (black dashed line)

## Understanding Your Results

### Good Training Signs

✅ **Episode reward increasing** - Agent learning to complete tasks
✅ **Collision rate decreasing** - Agent learning safety
✅ **Entropy > 0** - Maintaining exploration (bounded rationality)
✅ **Losses stabilizing** - Policy converging

### Risk-Averse Behavior (τ=0.1-0.5)

Expected characteristics:
- Lower collision rates (0-5%)
- Smoother trajectories
- Vehicles yield to each other
- Slightly longer travel times
- High goal achievement rate

### Risk-Neutral Behavior (τ=10.0)

Expected characteristics:
- Higher collision rates (15-25%)
- More aggressive maneuvers
- Faster travel times (when successful)
- Lower goal achievement rate

## Example Results

Your `traffic_merge_tau0.1_eps0.01` experiment shows:

```
Configuration:
  Scenario: merge
  Vehicles: 3
  Risk aversion (tau): 0.1 (very risk-averse)
  Bounded rationality (epsilon): 0.01

Evaluation Results (3 episodes):
  Mean Reward: 1426.08 ± 0.00
  Mean Collisions: 0.00
  Mean Goals Reached: 1.00
  Collision Rate: 0.0%
  Goal Rate: 33.3%
```

This shows **excellent safety** - no collisions across all episodes! The agent learned very conservative behavior.

## Tensorboard (Alternative)

For real-time monitoring during training:

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to see:
- All metrics in real-time
- Smoothed curves
- Multiple experiments compared side-by-side

## Command-Line Options

```bash
python -m src.visualize --help
```

Key options:
- `--exp_name`: Experiment directory name (required)
- `--log_dir`: Base log directory (default: "logs")
- `--render`: Render episodes with trained policy
- `--n_episodes`: Number of episodes to render (default: 10)
- `--save_fig`: Save figures instead of displaying

## Comparing Multiple Experiments

To compare different τ values:

```bash
# Train with different risk-aversion levels
./scripts/train_traffic.sh intersection 0.1 0.01  # Very risk-averse
./scripts/train_traffic.sh intersection 0.5 0.01  # Moderate
./scripts/train_traffic.sh intersection 10.0 0.01 # Risk-neutral

# Visualize each
for tau in 0.1 0.5 10.0; do
    uv run python -m src.visualize \
        --exp_name traffic_intersection_tau${tau}_eps0.01 \
        --render --n_episodes 5 --save_fig
done
```

Then compare:
- Collision rates across experiments
- Reward trajectories
- Vehicle behaviors in trajectory plots

## Visualization Files Location

All figures are saved in your experiment directory:

```
logs/YOUR_EXP_NAME/
├── config.json                  # Experiment configuration
├── metrics.json                 # All logged metrics
├── training_curves.png          # Training progress
├── evaluation_results.png       # Evaluation summary
├── trajectory_ep1.png           # Episode 1 trajectory
├── trajectory_ep2.png           # Episode 2 trajectory
└── ...
```

## Tips

1. **Always use `--save_fig`** to save plots for later analysis
2. **Render at least 3-5 episodes** to see behavior variation
3. **Use tensorboard** for real-time monitoring during training
4. **Compare trajectories** between risk-averse and risk-neutral agents
5. **Check entropy** stays positive - if it drops to 0, bounded rationality isn't working

## Troubleshooting

**Q: "No data for train/X" messages?**
A: Normal - we plot both prefixed and unprefixed keys, some may not exist

**Q: All trajectories look identical?**
A: Using deterministic evaluation - this is expected for converged policies

**Q: Plots don't show?**
A: Use `--save_fig` to save instead of displaying (better for remote servers)

**Q: Can't load model?**
A: Ensure the experiment has checkpoints in `logs/EXP_NAME/checkpoints/`

## Next Steps

After visualizing:
1. Analyze safety vs. performance tradeoffs
2. Try different τ values to find optimal risk-aversion
3. Test on different scenarios (intersection, merge, passing)
4. Compare to baseline (τ=10.0 ≈ standard MAPPO)
5. Write up results for your research paper!

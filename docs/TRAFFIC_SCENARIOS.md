# Traffic Scenarios for RQE-MAPPO

Custom autonomous vehicle coordination scenarios for testing risk-averse MARL.

## Overview

We provide three traffic scenarios to demonstrate RQE-MAPPO's benefits in safety-critical multi-agent settings:

1. **Intersection Crossing** - 4-way intersection without traffic lights
2. **Highway Merging** - Vehicles merging onto a highway
3. **Narrow Road Passing** - Vehicles passing on a narrow road

## Scenarios

### 1. Intersection Crossing

**Setup**:
- 3-4 vehicles approach a 4-way intersection from different directions
- No traffic light - must coordinate to avoid collisions
- Each vehicle has a goal on the opposite side

**Challenge**:
- High collision risk at intersection center
- Requires coordination and yielding
- Trade-off between speed and safety

**Why RQE helps**:
- Risk-averse agents (low τ) yield more, reducing collisions
- Bounded rationality (ε) prevents deterministic patterns that opponents can exploit

**Usage**:
```bash
./scripts/train_traffic.sh intersection 0.5 0.01
```

**Expected behavior**:
- Risk-averse agents (τ=0.1): Very cautious, may deadlock
- Moderate risk-aversion (τ=0.5): Good balance
- Risk-neutral (τ=∞): More collisions, faster when successful

---

### 2. Highway Merging

**Setup**:
- Main highway lane with vehicles traveling at constant speed
- Merging lane with vehicles trying to enter main lane
- Must find gaps and merge safely

**Challenge**:
- Timing merges correctly
- Avoiding collisions with main lane traffic
- Balancing speed with safety

**Why RQE helps**:
- Risk-averse agents wait for larger gaps (safer merges)
- Reduces aggressive merging that causes collisions

**Usage**:
```bash
./scripts/train_traffic.sh merge 0.5 0.01
```

---

### 3. Narrow Road Passing

**Setup**:
- Single narrow road
- Vehicles traveling in opposite directions
- Must coordinate passing maneuvers

**Challenge**:
- Limited space for passing
- Head-on collision risk
- Requires precise timing and positioning

**Why RQE helps**:
- Risk-averse agents slow down and create space
- Bounded rationality adds variation (less predictable)

**Usage**:
```bash
./scripts/train_traffic.sh passing 0.5 0.01
```

---

## Metrics

### Safety Metrics
- **Collision Rate**: Percentage of episodes with collisions
- **Near-Miss Count**: Close calls (distance < 2 * collision_radius)
- **Safety Violations**: Unsafe maneuvers (e.g., running intersection)

### Performance Metrics
- **Goal Reached Rate**: Percentage of vehicles reaching their goals
- **Average Travel Time**: Time to reach goal
- **Average Speed**: Mean vehicle speed

### Risk-Reward Tradeoff
We expect to see:
- Lower τ → Fewer collisions, slower travel times
- Higher τ → More collisions, faster travel times (when successful)
- Sweet spot: τ ≈ 0.3-0.7 for good balance

## Environment Details

### Observation Space
Each vehicle observes (14-dimensional for 3 vehicles):
- Own position (x, y): normalized to [0, 1]
- Own velocity (vx, vy): normalized by max_speed
- Goal position (goal_x, goal_y): normalized to [0, 1]
- Relative position of others: (x_i - x, y_i - y) for each other vehicle
- Relative velocity of others: (vx_i - vx, vy_i - vy)

### Action Space
5 discrete actions:
- 0: No acceleration (coast)
- 1: Accelerate forward
- 2: Brake (decelerate)
- 3: Turn left
- 4: Turn right

### Reward Function
```python
reward = 0.0

if crashed:
    reward = -100.0  # Large penalty for collision
elif reached_goal:
    reward = +100.0  # Large reward for success
else:
    # Progress reward (closer to goal = better)
    reward = -distance_to_goal * 0.1

    # Time penalty (encourages efficiency)
    reward -= 0.1
```

### Physics
- Simple kinematic model
- Discrete-time updates (dt = 0.1)
- Maximum speed: 2.0 units/second
- Collision detection: Euclidean distance < collision_radius

## Training

### Quick Test (10K steps)
```bash
./scripts/quick_test_traffic.sh
```

### Full Training (500K steps)
```bash
# Risk-averse (tau=0.5)
./scripts/train_traffic.sh intersection 0.5 0.01

# Risk-neutral (tau=10.0, like standard MAPPO)
./scripts/train_traffic.sh intersection 10.0 0.01

# Very risk-averse (tau=0.1)
./scripts/train_traffic.sh intersection 0.1 0.01
```

### Custom Training
```bash
uv run python -m src.train_traffic \
    --scenario intersection \
    --n_vehicles 3 \
    --total_timesteps 500000 \
    --tau 0.5 \
    --epsilon 0.01 \
    --risk_measure entropic \
    --exp_name my_experiment
```

## Monitoring

### Tensorboard
```bash
tensorboard --logdir logs/
```

### Key Plots to Watch
1. **Episode Reward**: Should increase over time
2. **Collision Rate**: Should decrease for risk-averse agents
3. **Goal Rate**: Should increase over time
4. **Entropy**: Should stay positive (bounded rationality working)
5. **Actor/Critic Loss**: Should stabilize

## Experiments to Run

### Experiment 1: Risk-Reward Tradeoff
Train with different τ values and plot:
- X-axis: Risk aversion (τ)
- Y-axis: Collision rate (red), Goal rate (green), Avg reward (blue)

Expected: Clear tradeoff curve

### Experiment 2: Robustness Test
1. Train on intersection scenario with τ=0.5
2. Test on:
   - Larger intersection (grid_size=30 instead of 20)
   - More vehicles (n_vehicles=4 instead of 3)
   - Faster speeds (max_speed=3.0 instead of 2.0)

Compare RQE-MAPPO vs standard MAPPO degradation

### Experiment 3: Heterogeneous Agents
Mix agents with different τ:
- Agent 0: τ=0.1 (very risk-averse)
- Agent 1: τ=1.0 (moderate)
- Agent 2: τ=10.0 (risk-seeking)

Observe emergent behavior (risk-averse agents yield to risk-seeking ones?)

### Experiment 4: Comparison to Reward Shaping
Create baseline with shaped rewards:
```python
# Standard reward
reward = -distance_to_goal * 0.1 - 0.1

# Shaped reward (equivalent to τ=0.5?)
if crashed:
    reward = -10000  # Large penalty
else:
    reward = -distance_to_goal * 0.1 - 0.1
```

Compare generalization across scenarios

## Expected Results

### Collision Rates
| τ Value | Intersection | Merge | Passing |
|---------|--------------|-------|---------|
| 0.1 (very risk-averse) | 2-5% | 1-3% | 3-6% |
| 0.5 (moderate) | 5-10% | 3-7% | 7-12% |
| 1.0 (mild) | 10-15% | 8-12% | 12-18% |
| 10.0 (risk-neutral) | 15-25% | 12-20% | 20-30% |

### Training Time
- Intersection: ~2-3 hours (500K steps on CPU)
- Merge: ~2-3 hours
- Passing: ~2-3 hours

## Limitations

1. **Simplified Physics**: Not as realistic as SUMO
2. **Limited Scenarios**: Only 3 scenarios (SUMO has unlimited)
3. **Discrete Actions**: Real AVs use continuous control
4. **No Sensors**: Direct state observation (no cameras/lidar)
5. **Single Lane**: No multi-lane dynamics

## Future Extensions

1. **Add SUMO Integration**: Use real traffic simulator
2. **Continuous Actions**: Steering angle, acceleration
3. **Sensor Models**: Camera/lidar observations
4. **More Scenarios**: Roundabouts, parking, etc.
5. **Human Drivers**: Mix AI and scripted human-like agents
6. **Real Traffic Data**: Train on real intersection patterns

## Citation

If you use these scenarios, please cite:

```bibtex
@inproceedings{mazumdar2025rqe,
  title={Tractable Multi-Agent Reinforcement Learning through Behavioral Economics},
  author={Mazumdar, Eric and Panaganti, Kishan and Shi, Laixi},
  booktitle={ICLR},
  year={2025}
}
```

---

**Questions?** See README.md or CLAUDE.md for more details.

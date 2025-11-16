#!/bin/bash
# Test SUMO GUI with a simple scenario

export SUMO_HOME="/usr/share/sumo"

echo "Testing SUMO-GUI..."
echo "SUMO_HOME: $SUMO_HOME"
echo ""
echo "This will open the SUMO GUI with random traffic."
echo "You can watch vehicles moving through intersections in real-time."
echo ""

uv run python -c "
import sumo_rl
import os

print('Starting SUMO environment with GUI...')
print('Note: XQuartz must be installed and running for GUI to work')
print('      If GUI doesn\'t work, run without --use_gui flag')
print()

# Create simple intersection environment with GUI
env = sumo_rl.parallel_env(
    net_file='single-intersection/single-intersection.net.xml',
    route_file='single-intersection/single-intersection-vhvh.rou.xml',
    use_gui=True,
    num_seconds=200,
)

print(f'Environment created with agents: {env.agents}')
print()

# Run one episode with random actions
observations = env.reset()
agents = env.agents

print('Running simulation...')
print('Watch the SUMO GUI window!')
print()

done = False
step = 0

while not done:
    actions = {agent: env.action_space(agent).sample()
              for agent in agents if agent in observations}

    observations, rewards, dones, truncated, infos = env.step(actions)

    done = all(dones.values()) or all(truncated.values())
    step += 1

    if step % 50 == 0:
        print(f'Step {step}...')

env.close()

print()
print(f'Simulation complete! Ran for {step} steps.')
"

# SUMO Setup Guide

SUMO (Simulation of Urban MObility) is a traffic simulation package that allows realistic multi-vehicle scenarios.

## Option 1: Install SUMO (Recommended for Production)

### macOS

```bash
# Download from official website
# Visit: https://sumo.dlr.de/docs/Downloads.php
# Download the macOS dmg file

# Or build from source:
brew install cmake eigen xerces-c fox
git clone https://github.com/eclipse/sumo.git
cd sumo
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
sudo make install

# Set environment variable
export SUMO_HOME="/path/to/sumo"
```

### Linux (Ubuntu/Debian)

```bash
# Add repository
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update

# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variable
export SUMO_HOME="/usr/share/sumo"
```

### Windows

```bash
# Download installer from:
# https://sumo.dlr.de/docs/Downloads.php

# Set environment variable:
# SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo
```

### Verify Installation

```bash
# Check SUMO is installed
sumo --version

# Check SUMO_HOME is set
echo $SUMO_HOME
```

## Option 2: Use Simple Grid-Based Traffic Environment (No SUMO Required)

We've created a custom grid-based traffic environment that simulates autonomous vehicle coordination without requiring SUMO installation.

**Advantages**:
- ✅ No external dependencies
- ✅ Fast and lightweight
- ✅ Easy to modify scenarios
- ✅ Good for initial experiments

**Disadvantages**:
- ❌ Less realistic than SUMO
- ❌ Simplified physics
- ❌ Limited to our custom scenarios

See `src/envs/traffic_grid.py` for the implementation.

## Option 3: Use SUMO-RL with Docker (Easiest)

```bash
# Pull Docker image with SUMO pre-installed
docker pull eclipse/sumo:latest

# Run container with SUMO
docker run -it -v $(pwd):/workspace eclipse/sumo:latest

# Inside container, install SUMO-RL
pip install sumo-rl
```

## Testing SUMO-RL

Once SUMO is installed:

```python
import sumo_rl
import os

# Set SUMO_HOME if not already set
os.environ['SUMO_HOME'] = '/path/to/sumo'

# Create environment
env = sumo_rl.parallel_env(
    net_file='path/to/net.xml',
    route_file='path/to/route.xml',
    use_gui=False,
    num_seconds=1000
)

# Test
env.reset()
print("SUMO-RL working!")
```

## For This Project

We recommend:
1. **For quick experiments**: Use our custom traffic grid (Option 2)
2. **For publication**: Install SUMO properly (Option 1)
3. **For reproducibility**: Use Docker (Option 3)

The RQE-MAPPO implementation works with any of these options!

#!/bin/bash
# Record trained RQE-MAPPO agent in SUMO as GIF

EXP_NAME=${1:-sumo_single-intersection_tau0.5_eps0.01}
OUTPUT=${2:-agent.gif}

echo "Recording agent from experiment: $EXP_NAME"
echo "Output: $OUTPUT"
echo ""
echo "⚠️  Important: SUMO-GUI will open. Make sure it's visible on screen!"
echo "   XQuartz must be installed and running for GUI screenshots."
echo ""

export SUMO_HOME="/usr/share/sumo"

uv run python -m src.record_sumo \
    --exp_name $EXP_NAME \
    --output $OUTPUT \
    --n_episodes 1 \
    --fps 5

echo ""
echo "Done! Check $OUTPUT"

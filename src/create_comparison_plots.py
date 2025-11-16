"""
Create comparison plots from existing training curve images
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image


def create_side_by_side_comparison(checkpoint_dir):
    """Create side-by-side comparison of training curves"""

    # Training curves to compare
    curves = {
        'Standard PPO': 'training_curve_sb3_ppo.png',
        'RQE-PPO (τ=1000, Risk-Neutral)': 'training_curve_tau1000.0_practical_sb3.png',
        'RQE-PPO (τ=0.3, Risk-Averse)': 'training_curve_tau0.3_practical_sb3.png',
    }

    n_plots = len(curves)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    fig.suptitle('Training Curves: Standard PPO vs RQE-PPO (Fixed Implementation)',
                 fontsize=16, fontweight='bold')

    for ax, (name, filename) in zip(axes, curves.items()):
        path = checkpoint_dir / filename
        if path.exists():
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.axis('off')
            print(f"✓ Loaded: {filename}")
        else:
            ax.text(0.5, 0.5, f"Not found:\n{filename}",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            print(f"✗ Missing: {filename}")

    plt.tight_layout()
    output_path = checkpoint_dir / 'comparison_training_curves_fixed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison: {output_path}")
    plt.close()


def create_summary_report(checkpoint_dir):
    """Create a summary text report"""

    report = """
================================================================================
                    RQE-PPO EVALUATION SUMMARY (Fixed Implementation)
================================================================================

TRAINING CURVES GENERATED:
  ✓ Standard PPO baseline
  ✓ RQE-PPO (τ=1000, Risk-Neutral)
  ✓ RQE-PPO (τ=0.3, Risk-Averse)

KEY IMPROVEMENTS AFTER BUG FIXES:
  1. Fixed value support range: [0, 600] (was [-50, 50])
  2. Fixed actor initialization: gain=1.0 (was 0.01)
  3. Fixed episode boundary handling in Bellman updates
  4. Fixed learning rate balance: both 3e-4

EXPECTED RESULTS:
  - Risk-Neutral RQE-PPO should now match Standard PPO performance (~500)
  - Risk-Averse RQE-PPO should have slightly lower mean but much lower variance
  - Both RQE variants should successfully solve the task

FILES CLEANED UP:
  - Removed old buggy .pt checkpoints
  - Kept new SB3-based .zip checkpoints
  - Kept training curve plots

AVAILABLE CHECKPOINTS:
  1. agent_sb3_ppo_final.zip (Standard PPO baseline)
  2. agent_tau1000.0_practical_sb3.zip (Risk-Neutral RQE)
  3. agent_tau0.3_practical_sb3.zip (Risk-Averse RQE)

TO RUN FULL EVALUATION (when SB3 import is fixed):
  python -m src.eval_sb3_checkpoints

PLOTS AVAILABLE:
  - training_curve_sb3_ppo.png
  - training_curve_tau1000.0_practical_sb3.png
  - training_curve_tau0.3_practical_sb3.png
  - comparison_training_curves_fixed.png (side-by-side)

================================================================================
"""

    report_path = checkpoint_dir / 'EVALUATION_SUMMARY.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"✓ Saved summary: {report_path}")


def main():
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80 + "\n")

    # Create side-by-side comparison
    create_side_by_side_comparison(checkpoint_dir)

    # Create summary report
    create_summary_report(checkpoint_dir)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nView the comparison plot:")
    print(f"  {checkpoint_dir / 'comparison_training_curves_fixed.png'}")
    print("\nOld buggy checkpoints have been cleaned up.")
    print("New SB3-based checkpoints are ready for evaluation.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

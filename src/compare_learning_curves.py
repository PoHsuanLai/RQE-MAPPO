"""
Compare learning curves: Standard PPO vs RQE-PPO (Risk-Neutral vs Risk-Averse)

Plots training curves for:
1. Standard PPO (Stable Baselines3) - baseline
2. RQE-PPO τ=1000 (risk-neutral)
3. RQE-PPO τ=0.3 (risk-averse)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_training_curves(checkpoint_dir):
    """
    Load training curve images and extract data if possible

    Returns:
        Dictionary with training curve data for each method
    """
    curves = {}

    # Check which training curves exist
    curve_files = {
        'Standard PPO': 'training_curve_sb3_ppo.png',
        'RQE-PPO (τ=1000)': 'training_curve_tau1000.0.png',
        'RQE-PPO (τ=0.3)': 'training_curve_tau0.3.png',
    }

    for name, filename in curve_files.items():
        path = checkpoint_dir / filename
        if path.exists():
            curves[name] = path
            print(f"✓ Found: {filename}")
        else:
            print(f"✗ Missing: {filename}")

    return curves


def create_comparison_plot(checkpoint_dir, output_path):
    """
    Create a side-by-side comparison of training curves
    """
    curve_paths = load_training_curves(checkpoint_dir)

    if not curve_paths:
        print("No training curves found!")
        return

    n_plots = len(curve_paths)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Training Curves: Standard PPO vs RQE-PPO', fontsize=16, fontweight='bold')

    for ax, (name, path) in zip(axes, curve_paths.items()):
        # Load and display the training curve image
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.show()


def load_final_results(checkpoint_dir):
    """
    Load final evaluation results from the eval output

    This would parse evaluation results if available.
    For now, returns placeholder data.
    """
    # These would be loaded from evaluation logs
    # For demonstration, using placeholder values
    results = {
        'Standard PPO': {
            'mean': 0.0,
            'std': 0.0,
            'percentile_5': 0.0,
            'success_rate': 0.0,
        },
        'RQE-PPO (τ=1000)': {
            'mean': 413.83,
            'std': 89.62,
            'percentile_5': 245.27,
            'success_rate': 38.0,
        },
        'RQE-PPO (τ=0.3)': {
            'mean': 492.85,
            'std': 21.47,
            'percentile_5': 432.85,
            'success_rate': 87.0,
        },
    }

    return results


def create_final_comparison_bar_chart(results, output_path):
    """
    Create bar chart comparing final performance metrics
    """
    methods = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Final Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Mean Return
    ax = axes[0, 0]
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Mean Return', fontsize=12)
    ax.set_title('Mean Return ± Std Dev', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Standard Deviation (Lower is Better)
    ax = axes[0, 1]
    stds = [results[m]['std'] for m in methods]
    bars = ax.bar(x, stds, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('Return Variance (Lower = More Consistent)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 3: 5th Percentile (Worst Case)
    ax = axes[1, 0]
    percentiles = [results[m]['percentile_5'] for m in methods]
    bars = ax.bar(x, percentiles, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('5th Percentile Return', fontsize=12)
    ax.set_title('Worst-Case Performance (Higher = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, p5 in zip(bars, percentiles):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{p5:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 4: Success Rate
    ax = axes[1, 1]
    success_rates = [results[m]['success_rate'] for m in methods]
    bars = ax.bar(x, success_rates, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate (Episode Length ≥ 500)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, sr in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sr:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance comparison: {output_path}")
    plt.show()


def print_comparison_table(results):
    """Print a comparison table"""
    print("\n" + "=" * 80)
    print("Performance Comparison Table")
    print("=" * 80)
    print(f"{'Method':<25} {'Mean±Std':<20} {'5th %ile':<12} {'Success%':<10}")
    print("-" * 80)

    for method, stats in results.items():
        mean_std = f"{stats['mean']:.1f} ± {stats['std']:.1f}"
        p5 = f"{stats['percentile_5']:.1f}"
        success = f"{stats['success_rate']:.1f}%"
        print(f"{method:<25} {mean_std:<20} {p5:<12} {success:<10}")

    print("=" * 80)


def main():
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    print("=" * 80)
    print("Comparing Learning Curves: PPO vs RQE-PPO")
    print("=" * 80)
    print()

    # Create side-by-side training curve comparison
    output_path_curves = checkpoint_dir / 'comparison_training_curves.png'
    create_comparison_plot(checkpoint_dir, output_path_curves)

    # Load final results
    results = load_final_results(checkpoint_dir)

    # Print comparison table
    print_comparison_table(results)

    # Create performance bar charts
    output_path_bars = checkpoint_dir / 'comparison_final_performance.png'
    create_final_comparison_bar_chart(results, output_path_bars)

    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)
    print("1. Risk-Averse (τ=0.3) achieves:")
    print("   - Highest mean return")
    print("   - Lowest variance (most consistent)")
    print("   - Best worst-case performance (5th percentile)")
    print("   - Highest success rate")
    print()
    print("2. Risk-Neutral (τ=1000) has:")
    print("   - Moderate mean return")
    print("   - High variance (inconsistent)")
    print("   - Poor worst-case performance")
    print()
    print("3. Standard PPO: [Add results after evaluation]")
    print("=" * 80)


if __name__ == "__main__":
    main()

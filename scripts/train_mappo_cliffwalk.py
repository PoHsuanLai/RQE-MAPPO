#!/usr/bin/env python3
"""
Train TRUE MAPPO (Centralized Critic) on Cliff Walk Environment

This implementation uses RLlib with centralized critic for true MAPPO:
- Centralized Training: Each agent's critic sees ALL agents' observations
- Decentralized Execution: Actors use only local observations
- Individual Rewards: Each agent optimizes its own reward (not shared)

Key insight: Even with individual rewards, the centralized critic helps because
the other agent's position affects the dynamics (proximity changes stochasticity).
"""

import argparse
import sys
from pathlib import Path
import numpy as np

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.tf_utils import explained_variance

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.cliff_walk import CliffWalkEnv, env_creator, simulate_trajectory, visualize_trajectory, get_normalized_obs

torch, nn = try_import_torch()

# Constants
OTHER_AGENTS_OBS = "other_agents_obs"
NUM_AGENTS = 2

# =============================================================================
# Centralized Critic Model
# =============================================================================

class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Multi-agent model with centralized value function for MAPPO.

    - Actor: Uses only local observation π(a|o_i)
    - Critic: Sees all agents' observations V(o_1, o_2)

    Each agent has its own critic that predicts ITS expected return,
    but using information from all agents' observations.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        num_agents = custom_config.get("num_agents", NUM_AGENTS)
        hiddens = model_config.get("fcnet_hiddens", [256, 256])

        single_obs_dim = obs_space.shape[0]
        central_obs_dim = num_agents * single_obs_dim

        # Actor network (uses only local observation)
        self.actor = TorchFC(obs_space, action_space, num_outputs, model_config, name + "_actor")

        # Centralized critic network (sees all observations)
        critic_layers = []
        last_dim = central_obs_dim
        for hidden_dim in hiddens:
            critic_layers.append(SlimFC(last_dim, hidden_dim, activation_fn=nn.Tanh))
            last_dim = hidden_dim
        critic_layers.append(SlimFC(last_dim, 1))
        self.central_vf = nn.Sequential(*critic_layers)

        self._central_obs_dim = central_obs_dim

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Forward pass for actor (uses only local obs)."""
        model_out, _ = self.actor(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, all_agents_obs):
        """
        Centralized value function V_i(o_1, o_2).

        Args:
            all_agents_obs: Concatenated observations from all agents
                           Shape: [batch_size, num_agents * obs_dim]

        Returns:
            Value estimates: [batch_size]
        """
        return torch.reshape(self.central_vf(all_agents_obs), [-1])

    @override(TorchModelV2)
    def value_function(self):
        """Local value function (not used in centralized training)."""
        return self.actor.value_function()


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# =============================================================================
# Centralized Critic PPO Policy
# =============================================================================

def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Collect other agents' observations and compute advantages with central VF.

    This is where the "centralized" part happens during training:
    - Collect observations from all agents
    - Use centralized value function for advantage computation
    """

    if hasattr(policy, "compute_central_vf") and other_agent_batches is not None:
        # Collect observations from all agents
        if policy.config.get("enable_connectors", False):
            other_batches = [batch for _, _, batch in other_agent_batches.values()]
        else:
            other_batches = [batch for _, batch in other_agent_batches.values()]

        # Concatenate all agents' observations
        # Shape: [batch_size, num_agents * obs_dim]
        all_obs = [sample_batch[SampleBatch.CUR_OBS]]
        for other_batch in other_batches:
            all_obs.append(other_batch[SampleBatch.CUR_OBS])

        all_agents_obs = np.concatenate(all_obs, axis=1)
        sample_batch[OTHER_AGENTS_OBS] = all_agents_obs

        # Compute centralized value function predictions
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(all_agents_obs, policy.device)
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros
        obs_dim = sample_batch[SampleBatch.CUR_OBS].shape[1]
        sample_batch[OTHER_AGENTS_OBS] = np.zeros(
            (len(sample_batch[SampleBatch.CUR_OBS]), NUM_AGENTS * obs_dim),
            dtype=np.float32
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    # Compute advantages using GAE
    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    """Loss function that uses centralized value function."""
    # Save original value function
    vf_saved = model.value_function

    # Replace with central value function for loss computation
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[OTHER_AGENTS_OBS]
    )
    policy._central_value_out = model.value_function()

    # Compute PPO loss with centralized critic
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    """Report explained variance of the central value function."""
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    """Create a centralized critic PPO policy class."""

    class CCPPOTorchPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            return loss_with_central_critic(self, super(), model, dist_class, train_batch)

        @override(base)
        def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def stats_fn(self, train_batch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTorchPolicy


# Create the centralized critic PPO policy
CCPPOTorchPolicy = get_ccppo_policy(PPOTorchPolicy)


# =============================================================================
# Main Training Script
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRUE MAPPO (Centralized Critic) on Cliff Walk"
    )

    # Environment parameters
    parser.add_argument("--horizon", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--reward_scale", type=float, default=50.0,
                        help="Reward scaling factor")
    parser.add_argument("--corner_reward", type=float, default=0.0,
                        help="One-time corner reward for shaping")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use more deterministic environment")

    # Training parameters
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use")
    parser.add_argument("--train_batch_size", type=int, default=4000,
                        help="Training batch size")
    parser.add_argument("--sgd_minibatch_size", type=int, default=128,
                        help="SGD minibatch size")
    parser.add_argument("--num_sgd_iter", type=int, default=10,
                        help="Number of SGD iterations per training batch")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lambda_", type=float, default=0.95,
                        dest="lambda_", help="GAE lambda")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")

    # Network architecture
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions")

    # Experiment parameters
    parser.add_argument("--stop_timesteps", type=int, default=500000,
                        help="Total timesteps to train")
    parser.add_argument("--stop_reward", type=float, default=None,
                        help="Stop when mean reward reaches this value")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Checkpoint frequency (iterations)")
    parser.add_argument("--local_dir", type=str,
                        default="results/mappo_cliffwalk",
                        help="Directory to save results")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--separate_policies", action="store_true",
                        help="Use separate policies for each agent (default: shared)")
    parser.add_argument("--enable_collision", action="store_true",
                        help="Enable collision dynamics (agents push each other randomly on collision)")

    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to load (for visualization only)")

    return parser.parse_args()


def load_checkpoint_and_visualize(args):
    """Load a checkpoint and generate visualizations without training."""
    from glob import glob

    checkpoint_path = Path(args.load_checkpoint).resolve()

    print("=" * 70)
    print("Loading MAPPO Checkpoint for Visualization")
    print("=" * 70)
    print(f"Checkpoint path: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return

    # Determine output directory (parent of checkpoint or experiment dir)
    # Checkpoint paths are usually like: results/mappo_cliffwalk/EXP_NAME/PPO_env/checkpoint_XXX
    output_dir = checkpoint_path.parent
    exp_name = checkpoint_path.parent.parent.name if checkpoint_path.parent.parent.exists() else "MAPPO_CliffWalk"

    # If user provided a trial directory (not a specific checkpoint), find the best checkpoint
    if not (checkpoint_path / "algorithm_state.pkl").exists() and not (checkpoint_path / "rllib_checkpoint.json").exists():
        # Look for checkpoint directories within
        checkpoint_dirs = sorted(glob(str(checkpoint_path / "checkpoint_*")))
        if not checkpoint_dirs:
            # Maybe it's an experiment directory - look deeper
            trial_dirs = sorted(glob(str(checkpoint_path / "PPO_*")))
            if trial_dirs:
                # Use the most recent trial
                trial_dir = trial_dirs[-1]
                checkpoint_dirs = sorted(glob(str(Path(trial_dir) / "checkpoint_*")))
                if checkpoint_dirs:
                    checkpoint_path = Path(checkpoint_dirs[-1])
                    output_dir = Path(trial_dir)
                    exp_name = checkpoint_path.parent.parent.parent.name

        elif checkpoint_dirs:
            # Use the last checkpoint
            checkpoint_path = Path(checkpoint_dirs[-1])

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment name: {exp_name}")

    # Register environment before restoring (required by RLlib)
    register_env(
        "cliff_walk",
        lambda config: ParallelPettingZooEnv(env_creator(config)),
    )

    # Restore the trained algorithm
    print("\nRestoring algorithm from checkpoint...")
    try:
        algo = PPO.from_checkpoint(str(checkpoint_path))
        print("  Algorithm restored successfully!")
    except Exception as e:
        import traceback
        print(f"ERROR: Could not restore algorithm from checkpoint: {e}")
        traceback.print_exc()
        return

    # Get the policy - handle both old and new RLlib APIs
    policy = None
    model = None
    module = None
    try:
        # Try new API first (RLlib 2.x with learner groups)
        if hasattr(algo, 'learner_group') and algo.learner_group is not None:
            # Get module from learner
            learner = algo.learner_group._learner
            module_keys = list(learner._module.keys())
            policy_id = "shared_policy" if "shared_policy" in module_keys else module_keys[0]
            module = learner._module[policy_id]
            print(f"  Using module (new API): {policy_id}")
            # Check if module has our custom model's method
            if hasattr(module, 'central_value_function'):
                model = module
            else:
                # Default RLlib module - use it for trajectory but skip value viz
                print(f"  Note: Module is {type(module).__name__}, may not have central_value_function")

        # Also try to get policy via workers (for trajectory simulation)
        if hasattr(algo, 'workers') and algo.workers is not None:
            local_worker = algo.workers.local_worker()
            policy_ids = list(local_worker.get_policies_to_train())
            if not policy_ids:
                policy_ids = list(local_worker.policy_map.keys())
            print(f"  Available policy IDs from worker: {policy_ids}")
            policy_id = "shared_policy" if "shared_policy" in policy_ids else (policy_ids[0] if policy_ids else "policy_0")
            policy = local_worker.get_policy(policy_id)
            if policy is not None:
                model = policy.model
                print(f"  Using policy from worker: {policy_id}")
    except Exception as e:
        import traceback
        print(f"Warning: Issue getting policy/model: {e}")
        traceback.print_exc()

    if model is None and module is None and policy is None:
        print("ERROR: Could not get any model or policy")
        algo.stop()
        return

    # Generate value function visualization
    print("\nGenerating value function visualization...")
    viz_model = model if model is not None else module
    if viz_model is not None and hasattr(viz_model, 'central_value_function'):
        try:
            visualize_mappo_values_from_model(viz_model, output_dir, exp_name)
        except Exception as e:
            import traceback
            print(f"Could not generate value function plot: {e}")
            traceback.print_exc()
    else:
        print("  Skipping value function visualization (no central_value_function method)")

    # Generate trajectory visualization
    print("\nSimulating and visualizing trajectory...")
    if policy is not None:
        try:
            simulate_mappo_trajectory_from_policy(policy, output_dir, exp_name)
        except Exception as e:
            import traceback
            print(f"Could not generate trajectory plot: {e}")
            traceback.print_exc()
    elif module is not None:
        try:
            simulate_mappo_trajectory_from_module(module, output_dir, exp_name)
        except Exception as e:
            import traceback
            print(f"Could not generate trajectory plot: {e}")
            traceback.print_exc()
    else:
        print("  Skipping trajectory visualization (no policy or module)")

    algo.stop()

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")


def visualize_mappo_values_from_model(model, output_dir, exp_name):
    """Visualize learned value function from MAPPO model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx in range(2):
        ax = axes[agent_idx]
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation for both agents (normalized)
                if agent_idx == 0:
                    agent1_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                    agent2_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                else:
                    agent1_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                    agent2_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0

                # Create global observation for centralized critic
                global_obs = np.concatenate([agent1_obs, agent2_obs])
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0)

                with torch.no_grad():
                    value = model.central_value_function(global_obs_tensor)
                    values[r, c] = value.item()

        im = ax.imshow(values, cmap='RdYlGn', origin='upper')

        # Add value annotations
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    color = 'white' if values[r, c] < (values.min() + values.max()) / 2 else 'black'
                    ax.text(c, r, f'{values[r, c]:.1f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')

        # Mark cliff cells
        for (cr, cc) in cliff_cells:
            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                       fill=True, facecolor='black', edgecolor='white', linewidth=2))

        # Mark goals
        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        title = f'Agent {agent_idx+1} Value (Goal at {my_goal})'
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'MAPPO (Centralized Critic) - Learned Values\n{exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = Path(output_dir) / "value_function.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Value function saved to {plot_path}")


def simulate_mappo_trajectory_from_policy(policy, output_dir, exp_name):
    """Simulate and visualize trajectory using trained MAPPO policy."""
    # Create a fresh environment for simulation
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=100,
        return_joint_reward=False,
    )

    # Create policy function that uses the trained MAPPO policy
    def mappo_policy(obs, env):
        # Get normalized observations for both agents
        obs_all = [get_normalized_obs(env, i) for i in range(2)]

        actions = []
        for i in range(2):
            # Compute action using policy
            obs_tensor = torch.FloatTensor(obs_all[i]).unsqueeze(0)
            with torch.no_grad():
                # Forward pass through actor
                logits, _ = policy.model.forward({"obs": obs_tensor}, [], None)
                # Get greedy action
                action = logits.argmax(dim=-1).item()
            actions.append(action)

        return actions

    print("  Simulating trajectory with trained MAPPO policy...")
    trajectory = simulate_trajectory(env, mappo_policy, max_steps=100)

    # Visualize
    traj_path = Path(output_dir) / "trajectory.png"
    visualize_trajectory(
        trajectory,
        save_path=str(traj_path),
        title=f"MAPPO Trajectory - {exp_name}"
    )


def simulate_mappo_trajectory_from_module(module, output_dir, exp_name):
    """Simulate and visualize trajectory using trained MAPPO module (new RLlib API)."""
    # Create a fresh environment for simulation
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=100,
        return_joint_reward=False,
    )

    # Create policy function that uses the trained module
    def mappo_policy(obs, env):
        # Get normalized observations for both agents
        obs_all = [get_normalized_obs(env, i) for i in range(2)]

        actions = []
        for i in range(2):
            # Compute action using module
            obs_tensor = torch.FloatTensor(obs_all[i]).unsqueeze(0)
            with torch.no_grad():
                # Try different forward methods based on module type
                if hasattr(module, 'forward_inference'):
                    output = module.forward_inference({"obs": obs_tensor})
                    if isinstance(output, dict) and "action_dist_inputs" in output:
                        logits = output["action_dist_inputs"]
                    else:
                        logits = output
                elif hasattr(module, '_forward_inference'):
                    output = module._forward_inference({"obs": obs_tensor})
                    logits = output.get("action_dist_inputs", output)
                else:
                    # Fallback: try direct forward
                    output = module.forward({"obs": obs_tensor})
                    if isinstance(output, dict):
                        logits = output.get("action_dist_inputs", output.get("logits", list(output.values())[0]))
                    else:
                        logits = output

                # Get greedy action
                if isinstance(logits, torch.Tensor):
                    action = logits.argmax(dim=-1).item()
                else:
                    action = 0  # Fallback
            actions.append(action)

        return actions

    print("  Simulating trajectory with trained MAPPO module...")
    trajectory = simulate_trajectory(env, mappo_policy, max_steps=100)

    # Visualize
    traj_path = Path(output_dir) / "trajectory.png"
    visualize_trajectory(
        trajectory,
        save_path=str(traj_path),
        title=f"MAPPO Trajectory - {exp_name}"
    )


def main():
    args = parse_args()

    # Suppress Ray's verbose logging
    import logging
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)

    # Initialize Ray with excludes to avoid uploading large files
    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
        log_to_driver=False,
        runtime_env={
            "excludes": [
                "results/",
                "*.git/",
                "sumo-rl/",
                "risk-aware-rl/",
                "archived_*/",
                "*.pack",
            ]
        }
    )

    # Register the centralized critic model
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # Handle checkpoint loading mode (visualization only)
    if args.load_checkpoint:
        load_checkpoint_and_visualize(args)
        ray.shutdown()
        return

    # Register environment
    env_name = "cliff_walk"
    env_config = {
        "grid_size": (6, 6),
        "horizon": args.horizon,
        "reward_scale": args.reward_scale,
        "corner_reward": args.corner_reward,
        "deterministic": args.deterministic,
        "enable_collision": args.enable_collision,
    }

    register_env(
        env_name,
        lambda config: ParallelPettingZooEnv(env_creator(config)),
    )

    # Create dummy env to get observation/action spaces
    dummy_env = env_creator(env_config)
    obs_space = dummy_env.observation_space("agent_0")
    act_space = dummy_env.action_space("agent_0")

    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Agents: {dummy_env.possible_agents}")

    # Policy configuration
    policy_config = {
        "framework": "torch",
        "gamma": args.gamma,
        "lambda": args.lambda_,
        "use_gae": True,
        "model": {
            "custom_model": "cc_model",
            "custom_model_config": {
                "num_agents": NUM_AGENTS,
            },
            "fcnet_hiddens": args.hidden_dims,
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        },
    }

    if args.separate_policies:
        # Separate policies for each agent
        policies = {
            "policy_0": (CCPPOTorchPolicy, obs_space, act_space, policy_config),
            "policy_1": (CCPPOTorchPolicy, obs_space, act_space, policy_config),
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: f"policy_{agent_id.split('_')[1]}"
    else:
        # Shared policy (parameter sharing)
        policies = {
            "shared_policy": (CCPPOTorchPolicy, obs_space, act_space, policy_config),
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

    # Configure PPO
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.horizon,
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.sgd_minibatch_size,
            num_epochs=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            use_gae=True,
            use_critic=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    # Experiment name
    exp_name = args.exp_name or "TRUE_MAPPO_CliffWalk"

    # Stop conditions
    stop = {"timesteps_total": args.stop_timesteps}
    if args.stop_reward is not None:
        stop["episode_reward_mean"] = args.stop_reward

    # Print configuration
    print("=" * 70)
    print("Starting TRUE MAPPO (Centralized Critic) Training")
    print("=" * 70)
    print(f"Environment: Cliff Walk")
    print(f"  Grid size: (6, 6)")
    print(f"  Horizon: {args.horizon}")
    print(f"  Reward scale: {args.reward_scale}")
    print(f"  Corner reward: {args.corner_reward}")
    print(f"  Deterministic: {args.deterministic}")
    print(f"  Collision dynamics: {args.enable_collision}")
    print()
    print(f"Training:")
    print(f"  Workers: {args.num_workers}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Entropy coefficient: {args.entropy_coeff}")
    print(f"  Total timesteps: {args.stop_timesteps}")
    print()
    print(f"Policy: {'Separate' if args.separate_policies else 'Shared'}")
    print("=" * 70)
    print("✓ TRUE MAPPO: Using CENTRALIZED critic V(o_1, o_2)")
    print("✓ Centralized Training: Critic sees all agents' observations")
    print("✓ Decentralized Execution: Actors use only local observations")
    print("✓ Individual Rewards: Each agent optimizes its own reward")
    print("=" * 70)

    # Run training
    # Convert relative path to absolute for storage_path
    storage_path = str(Path(args.local_dir).resolve())

    results = tune.run(
        PPO,
        name=exp_name,
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        storage_path=storage_path,
        verbose=1,
    )

    # Extract training data and plot
    # Try both old and new metric names
    try:
        best_trial = results.get_best_trial("episode_reward_mean", mode="max")
    except Exception:
        try:
            best_trial = results.get_best_trial("env_runners/episode_return_mean", mode="max")
        except Exception:
            # Just get any trial
            best_trial = results.trials[0] if results.trials else None

    if best_trial:
        # Get the trial's result dataframe
        trial_df = results.get_dataframe()

        print(f"\nDataframe columns: {trial_df.columns.tolist()}")
        print(f"Dataframe shape: {trial_df.shape}")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Extract metrics from trial results
            # Handle both old and new column names (RLlib API changes)
            # Old: episode_reward_mean, episode_len_mean
            # New: env_runners/episode_return_mean, env_runners/episode_len_mean
            reward_candidates = ['episode_reward_mean', 'env_runners/episode_return_mean']
            length_candidates = ['episode_len_mean', 'env_runners/episode_len_mean']
            timestep_candidates = ['timesteps_total', 'num_env_steps_sampled_lifetime']

            reward_col = next((c for c in reward_candidates if c in trial_df.columns), None)
            length_col = next((c for c in length_candidates if c in trial_df.columns), None)
            timestep_col = next((c for c in timestep_candidates if c in trial_df.columns), None)

            print(f"Using reward column: {reward_col}")
            print(f"Using length column: {length_col}")
            print(f"Using timestep column: {timestep_col}")

            episode_rewards = trial_df[reward_col].dropna().values if reward_col else np.array([])
            episode_lengths = trial_df[length_col].dropna().values if length_col else np.array([])
            timesteps = trial_df[timestep_col].dropna().values if timestep_col else np.arange(len(episode_rewards))

            if len(episode_rewards) == 0:
                print(f"Warning: No episode rewards found. Available columns: {trial_df.columns.tolist()}")
                # Try to extract from env_runners columns
                for col in trial_df.columns:
                    if 'return' in col.lower() or 'reward' in col.lower():
                        print(f"  Potential reward column: {col}")

            window = min(10, len(episode_rewards) // 5) if len(episode_rewards) > 5 else 1

            # Plot 1: Episode reward mean over timesteps
            if len(episode_rewards) > 0:
                axes[0, 0].plot(timesteps[:len(episode_rewards)], episode_rewards, 'b-', alpha=0.8)
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Episode Reward Mean')
            axes[0, 0].set_title('Episode Reward Mean vs Timesteps')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Episode length over timesteps
            if len(episode_lengths) > 0:
                axes[0, 1].plot(timesteps[:len(episode_lengths)], episode_lengths, 'g-', alpha=0.8)
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Episode Length Mean')
            axes[0, 1].set_title('Episode Length Mean vs Timesteps')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Episode reward over training iterations
            iterations = range(len(episode_rewards))
            if len(episode_rewards) > 0:
                axes[1, 0].plot(iterations, episode_rewards, 'purple', alpha=0.8)
            axes[1, 0].set_xlabel('Training Iteration')
            axes[1, 0].set_ylabel('Episode Reward Mean')
            axes[1, 0].set_title('Episode Reward Mean vs Iterations')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Smoothed reward if enough data
            if len(episode_rewards) > window and window > 0:
                smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(len(episode_rewards)), episode_rewards, alpha=0.3, label='Raw')
                axes[1, 1].plot(range(window-1, len(episode_rewards)), smoothed, 'r-', linewidth=2, label=f'Smoothed (w={window})')
                axes[1, 1].set_xlabel('Training Iteration')
                axes[1, 1].set_ylabel('Episode Reward Mean')
                axes[1, 1].set_title('Smoothed Episode Reward')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            elif len(episode_rewards) > 0:
                axes[1, 1].plot(iterations, episode_rewards, 'r-', alpha=0.8)
                axes[1, 1].set_xlabel('Training Iteration')
                axes[1, 1].set_ylabel('Episode Reward Mean')
                axes[1, 1].set_title('Episode Reward')
                axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'MAPPO Training Results - {exp_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save to the experiment directory
            plot_path = Path(storage_path) / exp_name / "learning_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nLearning curves saved to {plot_path}")
            plt.close()

            # Save training data as numpy files
            if len(episode_rewards) > 0:
                np.save(Path(storage_path) / exp_name / "episode_rewards.npy", episode_rewards)
            if len(episode_lengths) > 0:
                np.save(Path(storage_path) / exp_name / "episode_lengths.npy", episode_lengths)
            if len(timesteps) > 0:
                np.save(Path(storage_path) / exp_name / "timesteps.npy", timesteps)

        except Exception as e:
            import traceback
            print(f"Could not plot learning curves: {e}")
            traceback.print_exc()

        # Visualize learned value function
        try:
            visualize_mappo_values(best_trial, storage_path, exp_name)
        except Exception as e:
            import traceback
            print(f"Could not plot value function: {e}")
            traceback.print_exc()

        # Simulate and visualize trajectory
        try:
            simulate_mappo_trajectory(best_trial, storage_path, exp_name)
        except Exception as e:
            import traceback
            print(f"Could not simulate/visualize trajectory: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Print best result
    if best_trial:
        print(f"\nBest trial: {best_trial.trial_id}")
        # Try both old and new metric names
        last_result = best_trial.last_result or {}
        reward = last_result.get('episode_reward_mean') or last_result.get('env_runners', {}).get('episode_return_mean', 'N/A')
        if isinstance(reward, (int, float)):
            print(f"  Best reward: {reward:.2f}")
        else:
            print(f"  Best reward: {reward}")
        print(f"  Checkpoint: {best_trial.checkpoint}")
        print(f"\nResults saved to: {storage_path}/{exp_name}")

    ray.shutdown()


def visualize_mappo_values(best_trial, storage_path, exp_name):
    """Visualize learned value function from MAPPO centralized critic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Restore the trained algorithm
    checkpoint = best_trial.checkpoint
    if checkpoint is None:
        print("No checkpoint available for value visualization")
        return

    print(f"Restoring from checkpoint: {checkpoint}")

    try:
        algo = PPO.from_checkpoint(checkpoint)
    except Exception as e:
        print(f"Could not restore algorithm from checkpoint: {e}")
        print("Skipping value function visualization")
        return

    # Get the policy
    try:
        policy_ids = algo.get_policy_ids() if hasattr(algo, 'get_policy_ids') else []
        print(f"Available policy IDs: {policy_ids}")
        policy_id = "shared_policy" if "shared_policy" in policy_ids else "policy_0"
        policy = algo.get_policy(policy_id)
        if policy is None:
            print(f"Policy {policy_id} not found")
            algo.stop()
            return
        model = policy.model
    except Exception as e:
        print(f"Could not get policy/model: {e}")
        algo.stop()
        return

    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx in range(2):
        ax = axes[agent_idx]
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation for both agents (normalized)
                if agent_idx == 0:
                    agent1_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                    agent2_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                else:
                    agent1_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                    agent2_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0

                # Create global observation for centralized critic
                global_obs = np.concatenate([agent1_obs, agent2_obs])
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0)

                with torch.no_grad():
                    value = model.central_value_function(global_obs_tensor)
                    values[r, c] = value.item()

        im = ax.imshow(values, cmap='RdYlGn', origin='upper')

        # Add value annotations
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    color = 'white' if values[r, c] < (values.min() + values.max()) / 2 else 'black'
                    ax.text(c, r, f'{values[r, c]:.1f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')

        # Mark cliff cells
        for (cr, cc) in cliff_cells:
            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                       fill=True, facecolor='black', edgecolor='white', linewidth=2))

        # Mark goals
        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        title = f'Agent {agent_idx+1} Value (Goal at {my_goal})'
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'MAPPO (Centralized Critic) - Learned Values\n{exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = Path(storage_path) / exp_name / "value_function.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Value function saved to {plot_path}")

    algo.stop()


def simulate_mappo_trajectory(best_trial, storage_path, exp_name):
    """Simulate and visualize trajectory using trained MAPPO policy."""
    # Restore the trained algorithm
    checkpoint = best_trial.checkpoint
    if checkpoint is None:
        print("No checkpoint available for trajectory simulation")
        return

    print(f"Restoring algorithm for trajectory simulation...")

    try:
        algo = PPO.from_checkpoint(checkpoint)
    except Exception as e:
        print(f"Could not restore algorithm from checkpoint: {e}")
        return

    # Get the policy
    try:
        policy_ids = algo.get_policy_ids() if hasattr(algo, 'get_policy_ids') else []
        policy_id = "shared_policy" if "shared_policy" in policy_ids else "policy_0"
        policy = algo.get_policy(policy_id)
        if policy is None:
            print(f"Policy {policy_id} not found")
            algo.stop()
            return
    except Exception as e:
        print(f"Could not get policy: {e}")
        algo.stop()
        return

    # Create a fresh environment for simulation
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=100,
        return_joint_reward=False,
    )

    # Create policy function that uses the trained MAPPO policy
    def mappo_policy(obs, env):
        # Get normalized observations for both agents
        obs_all = [get_normalized_obs(env, i) for i in range(2)]

        actions = []
        for i in range(2):
            # Compute action using policy
            obs_tensor = torch.FloatTensor(obs_all[i]).unsqueeze(0)
            with torch.no_grad():
                # Forward pass through actor
                logits, _ = policy.model.forward({"obs": obs_tensor}, [], None)
                # Get greedy action
                action = logits.argmax(dim=-1).item()
            actions.append(action)

        return actions

    print("Simulating trajectory with trained MAPPO policy...")
    trajectory = simulate_trajectory(env, mappo_policy, max_steps=100)

    # Visualize
    traj_path = Path(storage_path) / exp_name / "trajectory.png"
    visualize_trajectory(
        trajectory,
        save_path=str(traj_path),
        title=f"MAPPO Trajectory - {exp_name}"
    )

    algo.stop()


if __name__ == "__main__":
    main()

"""
Training script for RQE-MAPPO on SUMO environments

Usage:
    python -m src.train_sumo --net single-intersection --tau 0.5 --epsilon 0.01
"""

import argparse
import torch
import numpy as np
import os
from pathlib import Path

import sumo_rl

from src.algorithms import RQE_MAPPO, RQEConfig
from src.utils import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQE-MAPPO on SUMO")

    # SUMO Environment
    parser.add_argument("--net", type=str, default="single-intersection",
                        choices=["single-intersection", "2way-single-intersection",
                                "2x2grid", "3x3grid", "4x4-Lucas", "4x4loop"],
                        help="SUMO network")
    parser.add_argument("--route_variant", type=str, default="vhvh",
                        help="Route variant (e.g., vhvh, ns, ew)")
    parser.add_argument("--num_seconds", type=int, default=1000,
                        help="Simulation seconds per episode")
    parser.add_argument("--delta_time", type=int, default=5,
                        help="Seconds between actions")
    parser.add_argument("--use_gui", action="store_true",
                        help="Use SUMO GUI (slower, for debugging)")
    parser.add_argument("--yellow_time", type=int, default=2,
                        help="Yellow phase duration")
    parser.add_argument("--min_green", type=int, default=5,
                        help="Minimum green phase duration")
    parser.add_argument("--max_green", type=int, default=50,
                        help="Maximum green phase duration")

    # RQE parameters
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Risk aversion parameter")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Bounded rationality parameter")
    parser.add_argument("--risk_measure", type=str, default="entropic",
                        choices=["entropic", "cvar", "mean_variance"],
                        help="Risk measure")

    # Network
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="PPO epochs")
    parser.add_argument("--n_minibatches", type=int, default=4,
                        help="Minibatches")

    # PPO
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=25000)
    parser.add_argument("--save_interval", type=int, default=100000)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def create_sumo_env(args):
    """Create SUMO environment"""
    # Set SUMO_HOME if not set
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'

    # Get sumo_rl package path
    import sumo_rl
    sumo_rl_path = Path(sumo_rl.__file__).parent
    nets_path = sumo_rl_path / 'nets'

    # Construct file paths (absolute paths)
    # Handle special cases where directory name differs from file prefix
    if args.net == "4x4-Lucas":
        file_prefix = "4x4"
    else:
        file_prefix = args.net

    net_file = str(nets_path / args.net / f"{file_prefix}.net.xml")

    # Try to find route file with variant
    route_file_variant = nets_path / args.net / f"{file_prefix}-{args.route_variant}.rou.xml"

    if route_file_variant.exists():
        route_file = str(route_file_variant)
    else:
        # Try common route file patterns
        possible_routes = [
            nets_path / args.net / f"{file_prefix}.rou.xml",
            nets_path / args.net / f"{file_prefix}c1.rou.xml",
            nets_path / args.net / f"{file_prefix}-vhvh.rou.xml",
        ]

        route_file = None
        for r in possible_routes:
            if r.exists():
                route_file = str(r)
                break

        if route_file is None:
            # Just use first .rou.xml file found
            route_files = list((nets_path / args.net).glob("*.rou.xml"))
            if route_files:
                route_file = str(route_files[0])
            else:
                raise FileNotFoundError(f"No route file found in {nets_path / args.net}")

    print(f"Network file: {net_file}")
    print(f"Route file: {route_file}")

    # Use direct SumoEnvironment
    from sumo_rl.environment.env import SumoEnvironment

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.use_gui,
        num_seconds=args.num_seconds,
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn='diff-waiting-time',  # Reward based on waiting time reduction
    )

    return env


class SUMOEnvWrapper:
    """Wrapper to make SUMO env compatible with our training loop"""

    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.agents = None
        self.n_agents = 0
        self.obs_dim = 0
        self.action_dim = 0

    def reset(self):
        """Reset environment"""
        obs_dict = self.env.reset()

        # Get agents (traffic signals)
        self.agents = list(self.env.ts_ids)
        self.n_agents = len(self.agents)

        # Get dimensions
        first_agent = self.agents[0]
        self.obs_dim = len(obs_dict[first_agent])
        self.action_dim = self.env.traffic_signals[first_agent].action_space.n

        # Convert dict to tensor (n_agents, obs_dim)
        obs_list = [obs_dict[agent] for agent in self.agents]
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)

        return obs_tensor

    def step(self, actions):
        """Step environment

        Args:
            actions: Tensor of shape (n_agents,)

        Returns:
            observations: Tensor (n_agents, obs_dim)
            rewards: Tensor (n_agents,)
            done: bool
            info: dict
        """
        # Convert actions to dict
        actions_np = actions.cpu().numpy().astype(int)
        action_dict = {agent: int(actions_np[i]) for i, agent in enumerate(self.agents)}

        # Step environment (returns dicts)
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # Convert dicts to tensors (in consistent agent order)
        obs_list = [obs_dict[agent] for agent in self.agents]
        reward_list = [reward_dict[agent] for agent in self.agents]

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device)

        # Episode done if all done
        done = all(done_dict.values())

        # Aggregate info
        info = {
            'avg_waiting_time': np.mean([sum(self.env.traffic_signals[ts_id].get_accumulated_waiting_time_per_lane())
                                        for ts_id in self.agents]),
            'system_total_waiting_time': info_dict.get('step_time', 0),
        }

        return obs_tensor, reward_tensor, done, info

    def close(self):
        self.env.close()


class RolloutBuffer:
    """Simple on-policy rollout buffer"""

    def __init__(self, buffer_size, n_agents, obs_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.device = device
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_observations = []
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, next_obs):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_observations.append(next_obs)
        self.ptr += 1

    def get(self):
        return {
            'observations': torch.stack(self.observations),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.stack(self.rewards),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=self.device),
            'next_observations': torch.stack(self.next_observations),
        }

    def __len__(self):
        return self.ptr


def train(args):
    """Main training loop"""

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    print("\nCreating SUMO environment...")
    sumo_env = create_sumo_env(args)
    env = SUMOEnvWrapper(sumo_env, device=device)

    # Initialize to get dimensions
    _ = env.reset()

    print(f"Environment: {args.net}")
    print(f"Agents: {env.n_agents}")
    print(f"Obs dim: {env.obs_dim}")
    print(f"Action dim: {env.action_dim}\n")

    # Create config
    config = RQEConfig(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        tau=args.tau,
        epsilon=args.epsilon,
        risk_measure=args.risk_measure,
        hidden_dims=args.hidden_dims,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches
    )

    # Create agent
    agent = RQE_MAPPO(config).to(device)

    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=args.batch_size,
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        device=device
    )

    # Create logger
    if args.exp_name is None:
        args.exp_name = f"sumo_{args.net}_tau{args.tau}_eps{args.epsilon}_seed{args.seed}"

    logger = Logger(
        log_dir=args.log_dir,
        exp_name=args.exp_name
    )

    # Save config
    logger.save_config(vars(args))

    # Training loop
    obs = env.reset()
    episode_reward = torch.zeros(env.n_agents, device=device)
    episode_length = 0
    episode_count = 0
    total_waiting_time = 0

    print(f"\n{'='*70}")
    print(f"Training RQE-MAPPO on SUMO: {args.net}")
    print(f"{'='*70}\n")

    for timestep in range(args.total_timesteps):
        # Collect experience
        with torch.no_grad():
            actions, log_probs, entropies = agent.get_actions(obs.unsqueeze(0))
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)

        # Step
        next_obs, rewards, done, info = env.step(actions)

        # Store
        buffer.add(obs, actions, log_probs, rewards, done, next_obs)

        # Update stats
        episode_reward += rewards
        episode_length += 1
        total_waiting_time += info['avg_waiting_time']

        # Episode end
        if done:
            # Log episode
            logger.log_scalar("episode/reward", episode_reward.mean().item(), timestep)
            logger.log_scalar("episode/length", episode_length, timestep)
            logger.log_scalar("episode/avg_waiting_time", total_waiting_time / episode_length, timestep)

            if (episode_count + 1) % 10 == 0:
                print(f"Episode {episode_count + 1} | Steps: {episode_length} | "
                      f"Reward: {episode_reward.mean().item():.2f} | "
                      f"Avg Wait: {total_waiting_time / episode_length:.1f}s")

            # Reset
            obs = env.reset()
            episode_reward = torch.zeros(env.n_agents, device=device)
            episode_length = 0
            total_waiting_time = 0
            episode_count += 1

        else:
            obs = next_obs

        # Update policy
        if len(buffer) >= args.batch_size:
            data = buffer.get()

            stats = agent.update(
                observations=data['observations'],
                actions=data['actions'],
                old_log_probs=data['log_probs'],
                rewards=data['rewards'],
                dones=data['dones'],
                next_observations=data['next_observations']
            )

            # Log training metrics
            for key, value in stats.items():
                logger.log_scalar(f"train/{key}", value, timestep)

            if (timestep // args.batch_size) % 10 == 0:
                print(f"\n[Update @ {timestep}]")
                print(f"  Actor Loss: {stats['actor_loss']:.4f}")
                print(f"  Critic Loss: {stats['critic_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}\n")

            buffer.clear()

        # Save model
        if (timestep + 1) % args.save_interval == 0:
            save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"model_{timestep+1}.pt"
            agent.save(str(save_path))
            print(f"Model saved: {save_path}")

    # Final save
    save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "model_final.pt"
    agent.save(str(final_path))

    logger.close()
    env.close()

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

"""
Custom Traffic Grid Environment for Autonomous Vehicle Coordination

A lightweight alternative to SUMO for testing RQE-MAPPO on traffic scenarios.

Features:
- Grid-based road network
- Multiple autonomous vehicles
- Collision detection
- Goal-reaching objectives
- Adjustable traffic density and aggression

Scenarios:
1. Intersection crossing (4-way, no traffic light)
2. Highway merging
3. Narrow road passing
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import gymnasium as gym


@dataclass
class VehicleState:
    """State of a single vehicle"""
    x: float  # X position
    y: float  # Y position
    vx: float  # X velocity
    vy: float  # Y velocity
    goal_x: float  # Goal X position
    goal_y: float  # Goal Y position
    crashed: bool = False
    reached_goal: bool = False


class TrafficGridEnv:
    """
    Multi-agent traffic environment

    Observation (per agent):
        - Own position (x, y)
        - Own velocity (vx, vy)
        - Goal position (goal_x, goal_y)
        - Relative positions of other vehicles (x_i - x, y_i - y) for all i
        - Relative velocities of other vehicles (vx_i - vx, vy_i - vy)

    Actions (discrete):
        - 0: No acceleration
        - 1: Accelerate forward
        - 2: Brake
        - 3: Turn left
        - 4: Turn right
    """

    def __init__(
        self,
        n_vehicles: int = 3,
        grid_size: float = 20.0,
        max_steps: int = 100,
        collision_radius: float = 0.5,
        goal_radius: float = 0.5,
        max_speed: float = 2.0,
        scenario: str = "intersection",
        device: str = "cpu"
    ):
        """
        Args:
            n_vehicles: Number of vehicles
            grid_size: Size of the grid
            max_steps: Maximum steps per episode
            collision_radius: Radius for collision detection
            goal_radius: Radius for goal reaching
            max_speed: Maximum vehicle speed
            scenario: "intersection", "merge", or "passing"
            device: torch device
        """
        self.n_vehicles = n_vehicles
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.collision_radius = collision_radius
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.scenario = scenario
        self.device = device

        # Action space: 5 discrete actions
        self.action_dim = 5

        # Observation space: own state (6) + other vehicles (4 * (n_vehicles - 1))
        self.obs_dim = 6 + 4 * (n_vehicles - 1)

        # State
        self.vehicles: List[VehicleState] = []
        self.step_count = 0

        print(f"TrafficGridEnv: {scenario}")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Grid size: {grid_size}")
        print(f"  Obs dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")

    def reset(self) -> torch.Tensor:
        """Reset environment"""
        self.step_count = 0
        self.vehicles = []

        # Initialize vehicles based on scenario
        if self.scenario == "intersection":
            self._init_intersection()
        elif self.scenario == "merge":
            self._init_merge()
        elif self.scenario == "passing":
            self._init_passing()
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        return self._get_observations()

    def _init_intersection(self):
        """Initialize 4-way intersection scenario"""
        # Vehicles approach from 4 directions
        positions = [
            (0.0, 10.0, 1.0, 0.0, 20.0, 10.0),  # From left
            (20.0, 10.0, -1.0, 0.0, 0.0, 10.0),  # From right
            (10.0, 0.0, 0.0, 1.0, 10.0, 20.0),  # From bottom
            (10.0, 20.0, 0.0, -1.0, 10.0, 0.0),  # From top
        ]

        for i in range(min(self.n_vehicles, 4)):
            x, y, vx, vy, gx, gy = positions[i]
            self.vehicles.append(VehicleState(x, y, vx, vy, gx, gy))

        # If more vehicles, add random ones
        for i in range(4, self.n_vehicles):
            pos = positions[i % 4]
            x, y, vx, vy, gx, gy = pos
            # Add some randomness
            x += np.random.randn() * 2.0
            y += np.random.randn() * 2.0
            self.vehicles.append(VehicleState(x, y, vx, vy, gx, gy))

    def _init_merge(self):
        """Initialize highway merge scenario"""
        # Main highway: vehicles at y=10
        # Merging lane: vehicles at y=5
        for i in range(self.n_vehicles):
            if i % 2 == 0:
                # Main highway
                x = i * 5.0
                y = 10.0
                vx = 1.0
                vy = 0.0
                gx = 20.0
                gy = 10.0
            else:
                # Merging lane
                x = i * 5.0
                y = 5.0
                vx = 1.0
                vy = 0.5  # Moving towards main lane
                gx = 20.0
                gy = 10.0

            self.vehicles.append(VehicleState(x, y, vx, vy, gx, gy))

    def _init_passing(self):
        """Initialize narrow road passing scenario"""
        # Narrow road: vehicles moving in opposite directions
        for i in range(self.n_vehicles):
            if i % 2 == 0:
                # Moving right
                x = 0.0
                y = 9.0 + np.random.randn() * 0.5
                vx = 1.0
                vy = 0.0
                gx = 20.0
                gy = 10.0
            else:
                # Moving left
                x = 20.0
                y = 11.0 + np.random.randn() * 0.5
                vx = -1.0
                vy = 0.0
                gx = 0.0
                gy = 10.0

            self.vehicles.append(VehicleState(x, y, vx, vy, gx, gy))

    def step(
        self,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        """
        Take a step

        Args:
            actions: [n_vehicles] discrete actions

        Returns:
            observations: [n_vehicles, obs_dim]
            rewards: [n_vehicles]
            done: bool
            info: dict
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Update vehicle states based on actions
        for i, (vehicle, action) in enumerate(zip(self.vehicles, actions)):
            if vehicle.crashed or vehicle.reached_goal:
                continue

            # Apply action
            accel = 0.1
            turn_rate = 0.2

            if action == 0:  # No acceleration
                pass
            elif action == 1:  # Accelerate forward
                speed = np.sqrt(vehicle.vx ** 2 + vehicle.vy ** 2)
                if speed > 0.1:
                    vehicle.vx += accel * vehicle.vx / speed
                    vehicle.vy += accel * vehicle.vy / speed
            elif action == 2:  # Brake
                vehicle.vx *= 0.9
                vehicle.vy *= 0.9
            elif action == 3:  # Turn left
                new_vx = vehicle.vx * np.cos(turn_rate) - vehicle.vy * np.sin(turn_rate)
                new_vy = vehicle.vx * np.sin(turn_rate) + vehicle.vy * np.cos(turn_rate)
                vehicle.vx = new_vx
                vehicle.vy = new_vy
            elif action == 4:  # Turn right
                new_vx = vehicle.vx * np.cos(-turn_rate) - vehicle.vy * np.sin(-turn_rate)
                new_vy = vehicle.vx * np.sin(-turn_rate) + vehicle.vy * np.cos(-turn_rate)
                vehicle.vx = new_vx
                vehicle.vy = new_vy

            # Clip speed
            speed = np.sqrt(vehicle.vx ** 2 + vehicle.vy ** 2)
            if speed > self.max_speed:
                vehicle.vx *= self.max_speed / speed
                vehicle.vy *= self.max_speed / speed

            # Update position
            vehicle.x += vehicle.vx * 0.1  # dt = 0.1
            vehicle.y += vehicle.vy * 0.1

            # Check boundaries
            vehicle.x = np.clip(vehicle.x, 0, self.grid_size)
            vehicle.y = np.clip(vehicle.y, 0, self.grid_size)

        # Check collisions
        for i in range(self.n_vehicles):
            for j in range(i + 1, self.n_vehicles):
                vi, vj = self.vehicles[i], self.vehicles[j]
                dist = np.sqrt((vi.x - vj.x) ** 2 + (vi.y - vj.y) ** 2)
                if dist < self.collision_radius:
                    vi.crashed = True
                    vj.crashed = True

        # Check goal reaching
        for vehicle in self.vehicles:
            dist_to_goal = np.sqrt(
                (vehicle.x - vehicle.goal_x) ** 2 +
                (vehicle.y - vehicle.goal_y) ** 2
            )
            if dist_to_goal < self.goal_radius:
                vehicle.reached_goal = True

        # Compute rewards
        rewards = self._compute_rewards()

        # Check if episode is done
        self.step_count += 1
        done = (
            self.step_count >= self.max_steps or
            all(v.crashed or v.reached_goal for v in self.vehicles)
        )

        # Get observations
        observations = self._get_observations()

        # Info
        info = {
            "collisions": sum(v.crashed for v in self.vehicles),
            "goals_reached": sum(v.reached_goal for v in self.vehicles),
            "step": self.step_count
        }

        return observations, rewards, done, info

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards for all vehicles"""
        rewards = []

        for vehicle in self.vehicles:
            reward = 0.0

            if vehicle.crashed:
                reward = -100.0  # Large penalty for collision
            elif vehicle.reached_goal:
                reward = 100.0  # Large reward for reaching goal
            else:
                # Progress towards goal
                dist_to_goal = np.sqrt(
                    (vehicle.x - vehicle.goal_x) ** 2 +
                    (vehicle.y - vehicle.goal_y) ** 2
                )
                reward = -dist_to_goal * 0.1

                # Small penalty for each step (encourages faster completion)
                reward -= 0.1

            rewards.append(reward)

        return torch.FloatTensor(rewards).to(self.device)

    def _get_observations(self) -> torch.Tensor:
        """Get observations for all vehicles"""
        observations = []

        for i, vehicle in enumerate(self.vehicles):
            # Own state: [x, y, vx, vy, goal_x, goal_y]
            own_state = [
                vehicle.x / self.grid_size,  # Normalize
                vehicle.y / self.grid_size,
                vehicle.vx / self.max_speed,
                vehicle.vy / self.max_speed,
                vehicle.goal_x / self.grid_size,
                vehicle.goal_y / self.grid_size,
            ]

            # Other vehicles: relative [x, y, vx, vy]
            other_states = []
            for j, other in enumerate(self.vehicles):
                if i != j:
                    other_states.extend([
                        (other.x - vehicle.x) / self.grid_size,
                        (other.y - vehicle.y) / self.grid_size,
                        (other.vx - vehicle.vx) / self.max_speed,
                        (other.vy - vehicle.vy) / self.max_speed,
                    ])

            obs = own_state + other_states
            observations.append(obs)

        return torch.FloatTensor(observations).to(self.device)

    def render(self):
        """Simple text-based rendering"""
        print(f"\nStep {self.step_count}")
        for i, vehicle in enumerate(self.vehicles):
            status = "CRASHED" if vehicle.crashed else "GOAL" if vehicle.reached_goal else "ACTIVE"
            print(
                f"Vehicle {i}: ({vehicle.x:.1f}, {vehicle.y:.1f}) "
                f"v=({vehicle.vx:.1f}, {vehicle.vy:.1f}) [{status}]"
            )


if __name__ == "__main__":
    # Test the environment
    print("Testing TrafficGridEnv...")

    env = TrafficGridEnv(
        n_vehicles=3,
        scenario="intersection"
    )

    obs = env.reset()
    print(f"Initial observations shape: {obs.shape}")

    # Run a few steps with random actions
    for step in range(10):
        actions = torch.randint(0, env.action_dim, (env.n_vehicles,))
        next_obs, rewards, done, info = env.step(actions)

        env.render()
        print(f"Rewards: {rewards.numpy()}")
        print(f"Info: {info}")

        if done:
            print("\nEpisode finished!")
            break

    print("\n✓ TrafficGridEnv test passed!")

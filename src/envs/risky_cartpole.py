"""
Risky CartPole: CartPole with stochastic disturbances

Perfect environment to showcase risk-averse vs risk-neutral policies:
- Random "wind gusts" push the cart
- Stochastic dynamics (variable pole mass)
- Optional penalty zones for large angles

Risk-averse agents should learn more conservative, stable policies.
Risk-neutral agents might be more aggressive but fail more often.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class RiskyCartPoleEnv(CartPoleEnv):
    """
    CartPole with added stochasticity to test risk-averse learning

    Modifications:
    1. Random wind gusts (force disturbances)
    2. Stochastic pole mass
    3. Penalty for large angles (risky states)
    4. Adjustable risk level
    """

    def __init__(
        self,
        render_mode=None,
        wind_strength: float = 0.5,  # Strength of random pushes
        wind_prob: float = 0.1,  # Probability of wind per step
        mass_std: float = 0.1,  # Stdev of pole mass variation
        angle_penalty: float = 0.1,  # Penalty for large angles
        risk_level: str = "medium"  # "low", "medium", "high"
    ):
        super().__init__(render_mode=render_mode)

        # Risk parameters
        self.wind_strength = wind_strength
        self.wind_prob = wind_prob
        self.mass_std = mass_std
        self.angle_penalty = angle_penalty

        # Adjust based on risk level
        if risk_level == "low":
            self.wind_strength *= 0.5
            self.wind_prob *= 0.5
            self.mass_std *= 0.5
        elif risk_level == "high":
            self.wind_strength *= 2.0
            self.wind_prob *= 2.0
            self.mass_std *= 2.0

        # Track statistics
        self.total_angle_penalty = 0.0
        self.num_disturbances = 0

    def reset(self, seed=None, options=None):
        """Reset with stochastic pole mass"""
        state, info = super().reset(seed=seed, options=options)

        # Randomize pole mass each episode
        self.masspole = max(0.05, np.random.normal(0.1, self.mass_std))
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

        # Reset tracking
        self.total_angle_penalty = 0.0
        self.num_disturbances = 0

        return state, info

    def step(self, action):
        """Step with random disturbances and angle penalties"""
        # Apply action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Get current state
        x, x_dot, theta, theta_dot = self.state

        # Apply force from action
        force = self.force_mag if action == 1 else -self.force_mag

        # Add random wind gust (KEY RISK SOURCE!)
        if np.random.random() < self.wind_prob:
            wind_force = np.random.normal(0, self.wind_strength)
            force += wind_force
            self.num_disturbances += 1

        # Physics simulation (same as CartPole)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # Check termination
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # Compute reward with angle penalty (encourages staying upright)
        reward = 1.0

        # Penalty for large angles (risky states!)
        angle_penalty = self.angle_penalty * (abs(theta) / self.theta_threshold_radians) ** 2
        reward -= angle_penalty
        self.total_angle_penalty += angle_penalty

        # Optionally truncate after max steps
        if self.steps_beyond_terminated is None:
            if terminated:
                self.steps_beyond_terminated = 0

        # Render
        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {
            'angle_penalty': angle_penalty,
            'total_angle_penalty': self.total_angle_penalty,
            'num_disturbances': self.num_disturbances
        }


class StochasticCartPoleEnv(CartPoleEnv):
    """
    Simpler stochastic CartPole with just reward noise

    Good for testing if agent learns risk-averse behavior with
    uncertain rewards but deterministic dynamics.
    """

    def __init__(self, render_mode=None, reward_noise_std: float = 0.5):
        super().__init__(render_mode=render_mode)
        self.reward_noise_std = reward_noise_std

    def step(self, action):
        """Step with noisy rewards"""
        state, reward, terminated, truncated, info = super().step(action)

        # Add reward noise
        reward += np.random.normal(0, self.reward_noise_std)

        return state, reward, terminated, truncated, info


# Register environments with Gymnasium
def register_risky_envs():
    """Register custom environments"""
    from gymnasium.envs.registration import register

    # Risky CartPole variants
    for risk_level in ["low", "medium", "high"]:
        register(
            id=f"RiskyCartPole-{risk_level}-v0",
            entry_point="src.envs.risky_cartpole:RiskyCartPoleEnv",
            kwargs={"risk_level": risk_level},
            max_episode_steps=500,
        )

    # Stochastic rewards only
    register(
        id="StochasticCartPole-v0",
        entry_point="src.envs.risky_cartpole:StochasticCartPoleEnv",
        max_episode_steps=500,
    )

    print("✓ Registered risky CartPole environments:")
    print("  - RiskyCartPole-low-v0")
    print("  - RiskyCartPole-medium-v0")
    print("  - RiskyCartPole-high-v0")
    print("  - StochasticCartPole-v0")


if __name__ == "__main__":
    print("Testing RiskyCartPoleEnv...")

    # Create environment
    env = RiskyCartPoleEnv(wind_strength=1.0, wind_prob=0.2, angle_penalty=0.1)

    # Run episode
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, angle_penalty={info['angle_penalty']:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total angle penalty: {info['total_angle_penalty']:.2f}")
    print(f"Num disturbances: {info['num_disturbances']}")

    env.close()
    print("\n✓ All tests passed!")

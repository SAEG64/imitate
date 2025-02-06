#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:20:36 2025

@author: sergej
"""
# %% ==========================================================================
# Generating custom env
# =============================================================================
import gymnasium.spaces as spaces
import gymnasium as gym
import numpy as np
import random

from game_paras import LP_init, gains, psucc

class ForestEnv(gym.Env):
    """Custom Foraging Environment where an agent must manage its energy to survive."""

    def __init__(self):
        super(ForestEnv, self).__init__()

        # Define action space
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(12,), dtype=np.float32)
        
        # Initialize environment parameters
        self.weather_prob_0 = None
        self.weather_prob_1 = None
        self.current_weather = None
        self.ternary_state = None
        self.horizon = None
        self.energy_gain_0 = None
        self.energy_gain_1 = None
        self.current_energy = None
        self.current_day = None
        self.done = None    # Terminal
        self.time = 5       # Constant sequence time

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # Set the random seed if provided
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Load env variables
        self.forest_draw = random.randint(0, len(psucc[0]) - 1)
        self.weather_prob_0 = np.float32([values[self.forest_draw] for key, values in psucc.items()][0])
        self.weather_prob_1 = np.float32([values[self.forest_draw] for key, values in psucc.items()][1])
        self.current_weather = np.random.choice([0, 1])  # Random initial weather
        self.ternary_state = 1  # Reset ternary state
        self.horizon = np.random.choice([3, 4, 5])       # Randomize forest duration
        self.energy_gain_0 = np.int32([values[self.forest_draw] for key, values in gains.items()][0])
        self.energy_gain_1 = np.int32([values[self.forest_draw] for key, values in gains.items()][1])
        self.current_energy = np.random.choice(LP_init)  # Randomize initial energy
        self.current_day = 0    # Reset current day
        self.done = np.float32(0)         # Reset 'done'
        # Additional info
        info = {
            'success': True if self.current_day >= self.time and self.current_energy != 0 else False}
        return self._get_obs(), info

    def step(self, action):
        """Apply an action and update the environment."""
        self.done = bool(self.done)
        if self.done:
            raise RuntimeError("Cannot step in a terminated environment. Reset it first.")
        
        # Ensure action is within valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get choice
        # p_fora = 1 / (1 + math.exp(-action[0])**4)
        choice = int(np.random.rand() < action)

        # Correct unequal horizons
        if self.current_day < self.horizon:
            # Apply choice: Wait or Forage
            if self.current_energy != 0:
                if choice == 0:  # Wait
                    self.current_energy -= 1  # Lose 1 energy
                elif choice == 1:  # Forage
                    if np.random.rand() < [self.weather_prob_0, self.weather_prob_1][int(self.current_weather)]:
                        outcome = [self.energy_gain_0, self.energy_gain_1][int(self.current_weather)]
                        self.current_energy += outcome
                    else:
                        self.current_energy -= 2  # Penalty for failed foraging
            else:
                self.current_energy = 0

        # Clip energy to the range [0, 5]
        self.current_energy = np.int32(max(0, min(self.current_energy, 5)))

        # Get reward
        self.reward = 0
        if self.current_energy == 0:
            self.reward -= 1  # Failure cost
        self.reward = np.float32(self.reward)

        # Update time-point and weather
        self.current_day += 1
        self.current_weather = np.random.choice([0, 1])
        # Update ternary state
        self.ternary_state = 1
        if self.current_energy == 1:
            self.ternary_state = 2
        elif self.current_energy > (
            self.horizon - self.current_day):
            self.ternary_state = 0
        # Termination when time is up
        if self.current_day >= self.time:
            self.done = True
        
        # Additional info
        info = {
            'success': True if self.current_day >= self.time and self.current_energy != 0 else False}

        return self._get_obs(), self.reward, self.done, False, info  # False is for "truncated"

    def render(self, mode='human'):
        """Render the current state of the environment."""
        obs = self._get_obs()
        weather = "Good" if self.current_weather == 1 else "Bad"
        print("==================================================")
        print(
            f"p_weather_0: {obs[0]}, p_weather_1: {obs[1]}")
        print(
            f"weather_0: {obs[2]}, weather_1: {obs[3]}")
        print(
            f"ternary_0: {obs[4]}, ternary_1: {obs[5]}, ternary_2: {obs[6]}")
        print(
            f"horizon: {obs[7]}, gain_weather_0: {obs[8]}, gain_weather_1: {obs[9]}")
        print(
            f"state: {obs[10]}, day: {obs[11]}")
        print("======")
        print(
            f"done: {self.done}")

    def _get_obs(self):
        """Get the current observation."""
        # One-hot encoder for weather and ternary state
        categ = [0, 0, 0, 0, 0]
        if self.current_energy != 0:
            categ[self.current_weather] = 1 # weather type
            categ[self.ternary_state+2] = 1 # ternary state

        return np.array([
            self.weather_prob_0,
            self.weather_prob_1,
            categ[0],
            categ[1],
            categ[2],
            categ[3],
            categ[4],
            self.horizon,
            self.energy_gain_0,
            self.energy_gain_1,
            self.current_energy,
            self.current_day], dtype=np.float32)
    
    def close(self):
        """Clean up resources if necessary."""
        pass

# # %% ==========================================================================
# # Testing custom env
# # =============================================================================
# env = ForestEnv()

# obs, _ = env.reset()
# done = False
# env.render()

# done_ls = []
# while not done:
#     action = np.array([np.random.uniform(0, 1)])[0]  # Sample random continuous action
#     print("======")
#     print("p_fora:", action)
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
#     done_ls.append(done)

# env.close()
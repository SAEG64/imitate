#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:20:36 2025

@author: sergej
"""

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random

from game_paras import LP_init, gains, psucc

class ForestEnv(gym.Env):
    """Custom Foraging Environment where an agent must manage its energy to survive."""

    def __init__(self):
        super(ForestEnv, self).__init__()

        # Define action space: 0 = Wait, 1 = Forage
        self.action_space = spaces.Discrete(2)

        # Define observation space:
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(8,), dtype=np.float32)  # Flattened Box space (all values as floats)

        # Initialize environment parameters
        self.current_day = None
        self.horizon = None
        self.current_energy = None
        self.weather_prob_0 = None
        self.weather_prob_1 = None
        self.energy_gain_0 = None
        self.energy_gain_1 = None
        self.current_weather = None
        # Terminal
        self.done = None
        # Constant sequence time
        self.time = 5

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # Set the random seed if provided
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Load env variables
        self.forest_draw = random.randint(0, len(psucc[0]) - 1)
        self.current_day = np.float32(0)  # Reset current day
        self.horizon = np.float32(np.random.choice([3, 4, 5]))       # Randomize forest duration
        self.current_energy = np.float32(np.random.choice(LP_init))  # Random initial energy
        self.weather_prob_0 = np.float32([values[self.forest_draw] for key, values in psucc.items()][0])
        self.weather_prob_1 = np.float32([values[self.forest_draw] for key, values in psucc.items()][1])
        self.energy_gain_0 = np.float32([values[self.forest_draw] for key, values in gains.items()][0])
        self.energy_gain_1 = np.float32([values[self.forest_draw] for key, values in gains.items()][1])
        self.current_weather = np.float32(np.random.choice([0, 1]))  # Random initial weather
        self.done = np.float32(0)         # Reset 'done'
        # Additional info
        info = {
            'days_left': self.horizon+3-self.current_day,
            'current p success': [self.weather_prob_0, self.weather_prob_1][int(self.current_weather)],
            "weather": ["bad", "good"][int(self.current_weather)]}
        return self._get_obs(), info

    def step(self, action):
        """Apply an action and update the environment."""
        self.done = bool(self.done)
        if self.done:
            raise RuntimeError("Cannot step in a terminated environment. Reset it first.")
        
        # Correct unequal horizons
        if self.current_day < self.horizon:
            # Apply action: Wait or Forage
            if self.current_energy != 0:
                if action == 0:  # Wait
                    self.current_energy -= 1  # Lose 1 energy
                elif action == 1:  # Forage
                    if np.random.rand() < [self.weather_prob_0, self.weather_prob_1][int(self.current_weather)]:
                        outcome = [self.energy_gain_0, self.energy_gain_1][int(self.current_weather)]
                        self.current_energy += outcome
                    else:
                        self.current_energy -= 2  # Penalty for failed foraging
            else:
                self.current_energy = 0

        # Clip energy to the range [0, 5]
        self.current_energy = max(0, min(self.current_energy, 5))

        # Update environment state
        self.current_day += 1
        if self.current_day < self.time:
            self.current_weather = np.random.choice([0, 1])  # Random new weather

        # Survival: reward agent if it survived the forest
        self.reward = 0
        if self.current_energy == 0:
            self.reward -= 1  # Failure cost
        # Enforce data type of rewards
        self.reward = np.float32(self.reward)
        
        # Check terminal conditions
        if self.current_day >= self.time:
            self.done = True
        
        # Additional info
        info = {
            'success': True if self.current_day >= self.time and self.current_energy != 0 else False}

        return self._get_obs(), self.reward, self.done, False, info  # False is for "truncated"

    def render(self, mode='human'):
        """Render the current state of the environment."""
        weather = "Good" if self.current_weather == 1 else "Bad"
        print(f"Day: {self.current_day}/{self.horizon+3}, Weather: {weather}, Energy: {self.current_energy}")

    def _get_obs(self):
        """Get the current observation."""
        return np.array([self.current_day,
                         self.horizon,
                         self.current_energy,
                         self.weather_prob_0, 
                         self.weather_prob_1, 
                         self.energy_gain_0, 
                         self.energy_gain_1, 
                         self.current_weather], dtype=np.float32)

    def close(self):
        """Clean up resources if necessary."""
        pass


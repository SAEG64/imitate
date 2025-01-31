#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# =============================================================================
# Preprocess data
# =============================================================================
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout#, types, wrappers
# from sklearn.preprocessing import OneHotEncoder
from typing import List
import pandas as pd
import numpy as np
import os
# Parse data
path = os.path.dirname(__file__)+"/"
os.chdir(path)
data = pd.read_csv("data_beh/datall_cat.csv")

# Data modifications
data['Reward'] = np.where(data['x12_continuous_energy_trial_end'] == 0, -1, 0).astype(np.float32)
data.loc[data['x14_p_foraging_gain'] > 1, 'x14_p_foraging_gain'] = 0
data['x7_weather_type'] = data['x7_weather_type']-1
data['x7_weather_type'] = data['x7_weather_type'].apply(lambda x: np.random.randint(0, 2) if x == -1 else x)

# encoder = OneHotEncoder()
# data[["State_Weather_Encoded"]] = encoder.fit_transform(data[["x7_weather_type"]]).toarray()

# Group data by Participant and Episode
grouped = data.groupby(["x1_id", "x4_index_forests", "x2_session"])

# Convert groups to trajectories
trajectories: List[TrajectoryWithRew] = []
Rs = []  # Raw rewards for evaluation
for _, group in grouped:
    group.reset_index(drop=True, inplace=True)
    # Observations, actions and rewards
    states = group[["x17_horizon_correct_adjusted", "x6_continuous_energy_trial_start",
                    "x59_weather_1_p_gain", "x60_weather_2_p_gain", 
                    "x57_weather_1_gain_magnitude", "x58_weather_2_gain_magnitude",
                    "x7_weather_type"]].values
    actions = group["x11_choice"].values
    rewards_raw = group["Reward"].values
    
    # Override "days left" with horizon
    repeated_entries = []
    states[:,0] = states[0,0]
    # Fake transitions for shorter trajectories
    if states.shape[0] < 5:
        # obs
        last_row = states[-1:]  # Select the last row
        repeat_count = 5 - states.shape[0]  # Measure how many copies
        repeated_rows = np.repeat(last_row, repeat_count, axis=0)   # Repeat last row (no change)
        repeated_rows[:, -1] = np.random.randint(0, 2, size=repeated_rows.shape[0]) # Fake weather draws
        states = np.vstack((states, repeated_rows))
        # act
        repeated_entries = np.full(repeat_count, np.random.randint(0, 2, size=1)[0])  # Repeat reward until reaching fake horizon
        actions = np.concatenate((actions, repeated_entries))  # Append to original array
        # rew
        last_entry = rewards_raw[-1]    # Get last reward
        repeated_entries = np.full(repeat_count, last_entry)  # Repeat reward until reaching fake horizon
        rewards = np.concatenate((rewards_raw, repeated_entries))  # Append to original array
        # info
        repeated_entries = [{'success': False} for x in range(repeat_count)]  # Last step of each episode is terminal

        
    # Add current day column
    row_count = states.shape[0]
    ascending_col = np.arange(1, row_count + 1).reshape(-1, 1) - 1
    states = np.hstack((ascending_col, states))
    
    # Add last day/outcome
    last_state = [5, group.iloc[0]["x17_horizon_correct_adjusted"],
                  group.iloc[-1]["x12_continuous_energy_trial_end"],
                  group.iloc[-1]["x59_weather_1_p_gain"], group.iloc[-1]["x60_weather_2_p_gain"], 
                  group.iloc[-1]["x57_weather_1_gain_magnitude"], group.iloc[-1]["x58_weather_2_gain_magnitude"],
                  np.random.randint(0, 2, size=1)[0]]
    states = np.append(states, [last_state], axis=0)
    # terminal = [True if data.iloc[x]['x17_horizon_correct_adjusted'] == 1 else False for x in range(len(data))]  # Last step of each episode is terminal
    done = [False] * (6 - 1) + [True]  # Last step of each episode is terminal
    # Additional information
    infos = repeated_entries + [{'success': True} if group.iloc[x]['x17_horizon_correct_adjusted'] == 1 and group.iloc[x]['x12_continuous_energy_trial_end'] != 0 else {'success': False} for x in range(len(group))]  # Last step of each episode is terminal
    # Assemble trajectory
    trajectories.append(TrajectoryWithRew(obs=states, acts=actions, rews=rewards, terminal=done, infos=infos))
    
    # Manually extract raw rewards (used for evaluation)
    Rs.append(rewards)
    
# Flatten trajectories
transitions = rollout.flatten_trajectories(trajectories)

# Aggregate data for evaluation
agg_R = [sum(Rs[i]) for i in range(len(Rs))]
agg = data.groupby(["x1_id", "x4_index_forests", "x2_session"]).sum()
agg['rewards'] = agg_R
agg = agg.groupby(["x1_id"]).mean()
# agg = agg.groupby(["x4_index_forests"]).mean()
agg_rew = agg['rewards']

# # Saving the trajectory data as a .npz file
# np.savez("data_beh/trajectories.npz", *[t for t in trajectories])

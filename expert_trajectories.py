#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# Preprocess data
# =============================================================================
from sklearn.preprocessing import OneHotEncoder
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

# Identify categorical and continuous features
categorical_features = ["x7_weather_type", "BNW_conditions"]
continuous_features = ["x17_horizon_correct_adjusted", "x6_continuous_energy_trial_start",
                "x59_weather_1_p_gain", "x60_weather_2_p_gain", 
                "x57_weather_1_gain_magnitude", "x58_weather_2_gain_magnitude",
                "x11_choice", "Reward", "x12_continuous_energy_trial_end"]

# Apply OneHotEncoder to categorical features
encoder = OneHotEncoder(drop=None, sparse_output=False)
encoded_categorical = encoder.fit_transform(data[categorical_features])
# Convert encoded features to DataFrame
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)
# Encode absorbing death state
encoded_df.loc[data['x6_continuous_energy_trial_start'] == 0, :] = 0

# Add categorical
processed_data = pd.concat([data[["x1_id", "x4_index_forests", "x2_session"]], data[continuous_features], encoded_df], axis=1)

# Group data by Participant and Episode
grouped = processed_data.groupby(["x1_id", "x4_index_forests", "x2_session"])

# %% ==========================================================================
# Encode trajectories
# =============================================================================
from imitation.data.types import TrajectoryWithRew
# from imitation.data.types import Transitions
from imitation.data import rollout
from typing import List
import torch

trajectories: List[TrajectoryWithRew] = []
Rs = []  # Raw rewards for evaluation
for _, group in grouped:
    group.reset_index(drop=True, inplace=True)

    # Observations, actions and rewards
    states = group[["x59_weather_1_p_gain", "x60_weather_2_p_gain"] +
                    list(encoded_df.columns) +
                    ["x17_horizon_correct_adjusted", 
                    "x57_weather_1_gain_magnitude", 
                    "x58_weather_2_gain_magnitude",
                    "x6_continuous_energy_trial_start"]].values    # Actions and rewards
    actions = group["x11_choice"].values
    rewards_raw = group["Reward"].values
    
    # Override "days left" with horizon
    repeated_entries = []
    states[:,-4] = states[0,-4]
    # Fake transitions for shorter trajectories
    if states.shape[0] < 5:
        # obs
        last_row = states[-1:]  # Select the last row
        repeat_count = 5 - states.shape[0]  # Measure how many copies
        repeated_rows = np.repeat(last_row, repeat_count, axis=0)   # Repeat last row (no change)
        if group.iloc[-1]["x12_continuous_energy_trial_end"] != 0:
            repeated_rows[:, np.random.randint(0, 2, size=1)[0]+2] = 1  # Random weather draws
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
    # Convert expert's binary action to continuous value between -1 and 1
    # actions = 2 * actions - 1
    actions = torch.tensor(actions, dtype=torch.float32).view(-1, 1)
    actions = np.array(actions, dtype=np.float32)
        
    # Add current day column
    row_count = states.shape[0]
    ascending_col = np.arange(1, row_count + 1).reshape(-1, 1) - 1
    states = np.hstack((states, ascending_col))

    ## Categorical variables
    categ = {col: 0 for col in encoded_df.columns}
    if group.iloc[-1]["x12_continuous_energy_trial_end"] != 0:
        # Encode random weather last day
        categ[list(categ.keys())[np.random.randint(0, 2, size=1)[0]]] = 1
        # Encode ternary state last day
        bnw = 1
        if group.iloc[-1]["x12_continuous_energy_trial_end"] == 1:
            bnw = 2
        elif group.iloc[-1]["x12_continuous_energy_trial_end"] > 1:
            bnw = 0
        categ[list(categ.keys())[bnw+2]] = 1 
    
    # Add last day/outcome
    last_state = [
        group.iloc[-1]["x59_weather_1_p_gain"], group.iloc[-1]["x60_weather_2_p_gain"]] + list(
        categ.values()) + [
        group.iloc[0]["x17_horizon_correct_adjusted"],
        group.iloc[-1]["x57_weather_1_gain_magnitude"], 
        group.iloc[-1]["x58_weather_2_gain_magnitude"],
        group.iloc[-1]["x12_continuous_energy_trial_end"]
        ] + [5]
    states = np.append(states, [last_state], axis=0)
    # states = states.astype(np.float32)

    # # Encode structured array with mixed continuous and discrete state spaces
    # dtype = [("col0", "f4"), ("col1", "f4")] + [(f"col{i}", "i4") for i in range(2, 12)]
    # structured_arr = np.zeros(states.shape[0], dtype=dtype)  # Initialize structured array
    # structured_arr["col0"] = states[:, 0]  # Assign first float column
    # structured_arr["col1"] = states[:, 1]  # Assign second float column
    # # Assign integer columns (converting from float to int)
    # for i in range(2, 12):
    #     structured_arr[f"col{i}"] = states[:, i].astype(np.int32)

    # ## Reshape to fit gym env obs_space
    # # Number of transitions
    # num_samples = states.shape[0]
    # # Reshape into (num_samples, 12)
    # obs_array = states.reshape((num_samples, 12))
    # # Split into components
    # obs_dict = {
    #     'weather_p_0': np.array(obs_array[:, 0], dtype=np.float32), 
    #     'weather_p_1': np.array(obs_array[:, 1], dtype=np.float32), 
    #     'weather_type_0': np.array(obs_array[:, 2], dtype=np.int32),    
    #     'weather_type_1': np.array(obs_array[:, 3], dtype=np.int32),    
    #     'ternary_state_0': np.array(obs_array[:, 4], dtype=np.int32),    
    #     'ternary_state_1': np.array(obs_array[:, 5], dtype=np.int32),    
    #     'ternary_state_2': np.array(obs_array[:, 6], dtype=np.int32),    
    #     'horizon': np.array(obs_array[:, 7], dtype=np.int32),   
    #     'weather_gain_0': np.array(obs_array[:, 8], dtype=np.int32),   
    #     'weather_gain_1': np.array(obs_array[:, 9], dtype=np.int32),   
    #     'energy_state': np.array(obs_array[:, 10], dtype=np.int32),
    #     'time_point': np.array(obs_array[:, 11], dtype=np.int32) 
    # }
    # # Reformat dict
    # # obs_nest = np.array([
    # #         [obs_dict[key][i] for key in obs_dict] for i in range(len(actions)+1)])
    # obs_list = [
    # {key: obs_dict[key][i] for key in obs_dict} for i in range(len(actions) + 1)]

    # terminal = [True if data.iloc[x]['x17_horizon_correct_adjusted'] == 1 else False for x in range(len(data))]  # Last step of each episode is terminal
    done = [False] * (len(actions)-1) + [True]  # Last step of each episode is terminal
    # Additional information
    infos = repeated_entries + [{'success': True} if group.iloc[x]['x17_horizon_correct_adjusted'] == 1 and group.iloc[x]['x12_continuous_energy_trial_end'] != 0 else {'success': False} for x in range(len(group))]  # Last step of each episode is terminal
    # Assemble trajectory
    trajectories.append(TrajectoryWithRew(obs=states, acts=actions, rews=rewards, terminal=done, infos=infos))
    
    # Manually extract raw rewards (used for evaluation)
    Rs.append(rewards)

transitions = rollout.flatten_trajectories(trajectories)

# # Manually flatten trajectories
# obs_list = []
# next_obs_list = []
# acts_list = []
# infos_list = []
# dones_list = []
# for traj in trajectories:
#     obs_list.extend(traj.obs[:-1])      # All observations except the last
#     next_obs_list.extend(traj.obs[1:])  # All observations except the first
#     acts_list.extend(traj.acts)         # Actions
#     infos_list.extend(traj.infos)         # Infos
#     dones_list.extend(traj.terminal[:])  # Dones (excluding last obs)
# # Convert list of dicts into dict of numpy arrays
# # obs_array = np.array([{key: obs[key]} for obs in obs_list]) for key in obs_list[0]}
# # next_obs_array = {key: np.array([obs[key] for obs in next_obs_list]) for key in next_obs_list[0]}
# # Convert lists to numpy arrays
# # obs_batch = np.stack([list(obs.values()) for obs in obs_list])
# # next_obs_batch = np.stack([list(obs.values()) for obs in next_obs_list])
# # obs_arrays = np.array(np.array(obs_list[i]) for i in range(len(obs_list)))
# # next_obs_arrays = np.array(np.array(next_obs_list[i]) for i in range(len(next_obs_list)))
# obs_array = np.array(obs_list)
# next_obs_array = np.array(next_obs_list)
# acts_array = np.array(acts_list)
# infos_array = np.array(infos_list)
# dones_array = np.array(dones_list)

# # torch.tensor(actions, dtype=torch.float32).view(-1, 1)

# # Convert lists to numpy arrays
# transitions = Transitions(
#     obs=obs_array,
#     acts=acts_array,
#     next_obs=obs_array, 
#     dones=dones_array, 
#     infos=infos_array
# )

# %% ==========================================================================
# Aggregate data for evaluation
# =============================================================================
agg_R = [sum(Rs[i]) for i in range(len(Rs))]
agg = data.groupby(["x1_id", "x4_index_forests", "x2_session"]).sum()
dec = data.groupby(["x1_id", "x4_index_forests", "x2_session"]).agg({'x11_choice': 'mean'})
agg['av_choice'] = dec['x11_choice']
agg['rewards'] = agg_R
agg = agg.groupby(["x1_id"]).mean()
agg_rew = agg['rewards']
agg_act = agg['av_choice']

# # Saving the trajectory data as a .npz file
# np.savez("data_beh/trajectories.npz", *[t for t in trajectories])

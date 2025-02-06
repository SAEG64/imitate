#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# Set up requirements
# =============================================================================
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
import pandas as pd
import numpy as np
import sys
import os

path = os.path.dirname(sys.argv[0]) + "/"
SEED = 42
rng = np.random.default_rng(SEED)

# %% ==========================================================================
# Set up environment
# =============================================================================
from gymnasium.envs.registration import register

# Register custom environment
register(
    id="ForagingEnv-v0",
    entry_point="foraging_env:ForestEnv",  # Update with the correct module path
)

# Create the environment
# This will prevent multiprocessing issues in Windows and other platforms
vec_env = make_vec_env(
    "ForagingEnv-v0",
    rng=rng,
    n_envs=5,                  # Number of parallel environments
    parallel=False,             # Use parallelism (SubprocVecEnv)
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)  # needed for computing rollouts later
    ]
)
    
# %% ==========================================================================
# Set up and run behavioral cloning
# =============================================================================
from expert_trajectories import transitions, agg_rew

batch_sizes = [32, 64, 128]


bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=transitions,
    batch_size=batch_sizes[0],
    rng=rng
)
bc_trainer.train(n_epochs=20)

# %% ==========================================================================
# Get state-action values
# =============================================================================
import torch

obs = torch.tensor(vec_env.reset(), dtype=torch.float32)
action, value, features = bc_trainer.policy.forward(obs)  # Get the raw features
print("Raw network features (before tanh or activation):", features)

# %% ==========================================================================
# Evaluate rewards and success rates
# =============================================================================
from stable_baselines3.common.evaluation import evaluate_policy

## Rewards
# Evaluate BC policy on the environment
reward_mean, reward_std = evaluate_policy(bc_trainer.policy, vec_env, n_eval_episodes=150)
# Compare BC to export
print(f"BC model predicted mean reward: {reward_mean:.2f} ± {reward_std:.2f}")
print(f"empirical aggregated mean reward: {np.mean(agg_rew):.2f} ± {np.std(agg_rew):.2f}")

## Success rates BC model
# Get success rate of BC model
mod_succ = []  # Initialize success list here
def success_callback(locals_, globals_):
    """ Custom callback to count successful episodes. """
    info = locals_['info']  # Get the latest step's info dictionary
    if "success" in info and info["success"]:
        globals()["mod_succ"].append(1)  # Track successful episodes
    else:
        globals()["mod_succ"].append(0)
# Evaluate the BC model
success_rate = evaluate_policy(
    bc_trainer.policy, vec_env, n_eval_episodes=120,
    return_episode_rewards=False,  # We only need success info
    deterministic=True,
    warn=False,
    callback=success_callback)
# Compare BC to export
# Get success rates from trajectories
emp_succ = [transitions.infos[i]['success'] for i in range(len(transitions))]
print(f"BC Model Success Rate: {sum(mod_succ):.2f}%")
print(f"empirical Success Rate: {sum(emp_succ)/27:.2f}%")

# %% ==========================================================================
# Evaluate actions distributions
# =============================================================================
actions_expert = transitions.acts  # Collect actions from the expert
# Get model actions and trajectories
actions_bc = []  # Collect actions from the BC model
traject_bc = []  # Collect states from BC trajectories
for _ in range(450):
    obs = vec_env.reset()
    for day in range(5):
        traject_bc = traject_bc + [str(list(obs[i])) for i in range(len(obs))]
        action, _ = bc_trainer.policy.predict(obs, deterministic=True)
        obs = vec_env.step(action)[0]
        actions_bc = actions_bc + list(action)

## Compare state-action pairs
# Get expert states
traject_expert = [str(list(transitions.obs[i])) for i in range(len(transitions.obs))]
# Aggregate state-actions for expert
pd_traj_ex = pd.DataFrame([traject_expert, actions_expert]).T
pd_traj_ex.columns = ['states', 'actions']
ag_traj_ex = pd_traj_ex.groupby(["states"]).mean()
# Aggregate state-actions for model
pd_traj_bc = pd.DataFrame([traject_bc, actions_bc]).T
pd_traj_bc.columns = ['states', 'actions']
ag_traj_bc = pd_traj_bc.groupby(["states"]).mean()
# Compare
trajs = pd.merge(ag_traj_ex, ag_traj_bc, left_index=True, right_index=True, how='left')
trajs.columns = ['expert', 'bc_model']
correlation = trajs[['expert', 'bc_model']].corr()
print(correlation)

# # Plot action distributions
# import matplotlib.pyplot as plt
# plt.hist(actions_bc, bins=20, alpha=0.5, label="BC Policy")
# plt.hist(actions_expert, bins=20, alpha=0.5, label="Expert Policy")
# plt.legend()
# plt.xlabel("Actions")
# plt.ylabel("Frequency")
# plt.title("Action Distributions: BC vs Expert")
# plt.show()
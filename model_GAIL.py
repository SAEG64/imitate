#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# Set up requirements
# =============================================================================
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import sys
import os

path = os.path.dirname(sys.argv[0]) + "/"
SEED = 42

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
    rng=np.random.default_rng(SEED),
    n_envs=5,                  # Number of parallel environments
    parallel=False,             # Use parallelism (SubprocVecEnv)
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)  # needed for computing rollouts later
    ]
)
    
# %% ==========================================================================
# Specify GAIL parameters
# =============================================================================
import torch.nn as nn

# Modify hidden layers
policy_kwargs = dict(
    net_arch=[32, 32],      # Reduce hidden layer size (default: [64, 64])
    activation_fn=nn.Tanh,   # Change activation function Tanh or ReLU
)

# Learning rate schedule
def linear_schedule(initial_value):
    """ Linear learning rate decay function. """
    def func(progress_remaining):
        return progress_remaining * initial_value  # Linear decay
    return func
def custom_lr_schedule(progress_remaining):
    """ Exponential learning rate decay function. """
    return 3e-4 * (0.95 ** (1 - progress_remaining))  # Exponential decay

# %% ==========================================================================
# Set up and train GAIL
# =============================================================================
from trajectories import transitions, agg_rew

# Initialize PPO as generator algorithm
gen_algo = PPO("MlpPolicy", vec_env, verbose=1,
    n_steps=2048,        # Adjust the number of steps per update
    batch_size=32,       # Reduce batch size for more frequent updates
    ent_coef=0.02,       # Encourage exploration
    policy_kwargs=policy_kwargs,
    # learning_rate=custom_lr_schedule(1),  # Decreasing learning rate
    learning_rate=linear_schedule(0.3),  # Decreasing learning rate
)

# Define a reward network for GAIL
reward_net = BasicRewardNet(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space
)

# Initialize GAIL using the expert trajectories
gail_trainer = GAIL(
    demonstrations=transitions,  # Expert demonstrations
    venv=vec_env,           # Vectorized environment
    demo_batch_size=64,     # Adjust based on available data
    gen_algo=gen_algo,      # Generator RL algorithm (PPO)
    reward_net=reward_net   # Reward model for imitation learning
)

# Train GAIL
gail_trainer.train(total_timesteps=100000)

# %% ==========================================================================
# Evaluate rewards and success rates
# =============================================================================
from stable_baselines3.common.evaluation import evaluate_policy

## Rewards
# Evaluate GAIL policy on the environment
reward_mean, reward_std = evaluate_policy(gail_trainer.policy, vec_env, n_eval_episodes=150)
# Compare GAIL to export
print(f"GAIL model predicted mean reward: {reward_mean:.2f} ± {reward_std:.2f}")
print(f"empirical aggregated mean reward: {np.mean(agg_rew):.2f}")# ± {np.std(agg_rew):.2f}")

## Success rates GAIL model
# Get success rate of GAIL model
mod_succ = []  # Initialize success list here
def success_callback(locals_, globals_):
    """ Custom callback to count successful episodes. """
    info = locals_['info']  # Get the latest step's info dictionary
    if "success" in info and info["success"]:
        globals()["mod_succ"].append(1)  # Track successful episodes
    else:
        globals()["mod_succ"].append(0)
# Evaluate the GAIL model
success_rate = evaluate_policy(
    gail_trainer.policy, vec_env, n_eval_episodes=120,
    return_episode_rewards=False,  # We only need success info
    deterministic=True,
    warn=False,
    callback=success_callback)
# Compare GAIL to export
# Get success rates from trajectories
emp_succ = [transitions.infos[i]['success'] for i in range(len(transitions))]
print(f"GAIL Model Success Rate: {sum(mod_succ):.2f}%")
print(f"empirical Success Rate: {sum(emp_succ)/27:.2f}%")

# %% ==========================================================================
# Evaluate actions distributions
# =============================================================================
import matplotlib.pyplot as plt

# Get model actions and trajectories
actions_gail = []  # Collect actions from the GAIL model
traject_gail = []  # Collect states from GAIL trajectories
for _ in range(450):
    obs = vec_env.reset()
    for day in range(5):
        traject_gail = traject_gail + [str(list(obs[i])) for i in range(len(obs))]
        action, _ = gail_trainer.policy.predict(obs, deterministic=True)
        obs = vec_env.step(action)[0]
        actions_gail = actions_gail + list(action)
    
## Compare action distributions
# Get expert actions
actions_expert = transitions.acts  # Collect actions from the expert
# Plot action distributions
plt.hist(actions_gail, bins=20, alpha=0.5, label="GAIL Policy")
plt.hist(actions_expert, bins=20, alpha=0.5, label="Expert Policy")
plt.legend()
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.title("Action Distributions: GAIL vs Expert")
plt.show()

## Compare state-action pairs
# Get expert states
traject_expert = [str(list(transitions.obs[i])) for i in range(len(transitions.obs))]
# Aggregate state-actions for expert
pd_traj_ex = pd.DataFrame([traject_expert, actions_expert]).T
pd_traj_ex.columns = ['states', 'actions']
ag_traj_ex = pd_traj_ex.groupby(["states"]).mean()
# Aggregate state-actions for model
pd_traj_gail = pd.DataFrame([traject_gail, actions_gail]).T
pd_traj_gail.columns = ['states', 'actions']
ag_traj_gail = pd_traj_gail.groupby(["states"]).mean()
# Compare
trajs = pd.merge(ag_traj_ex, ag_traj_gail, left_index=True, right_index=True, how='left')
trajs.columns = ['expert', 'GAIL_model']
correlation = trajs[['expert', 'GAIL_model']].corr()
print(correlation)
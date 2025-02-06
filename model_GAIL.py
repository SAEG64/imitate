#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# General requirements
# =============================================================================
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
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.envs.registration import register
from imitation.util.util import make_vec_env

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
# Specify DNNs
# =============================================================================
from imitation.rewards.reward_nets import BasicRewardNet
import torch.nn as nn

# Modify hidden layers
policy_kwargs = dict(
    net_arch=[64, 64],
    # pi=[256, 256],
    # vf=[256, 256],
    activation_fn=nn.Sigmoid,   # Activaation function
)

# Define a reward network for GAIL
reward_net = BasicRewardNet(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    # normalize_input_layer="running_norm",
    hid_sizes=[64],
    activation=nn.ReLU,   # Activaation function
)

# # Learning rate schedule
# def linear_schedule(initial_value):
#     """ Linear learning rate decay function. """
#     def func(progress_remaining):
#         return progress_remaining * initial_value  # Linear decay
#     return func
# def custom_lr_schedule(progress_remaining):
#     """ Exponential learning rate decay function. """
#     return 3e-4 * (0.95 ** (1 - progress_remaining))  # Exponential decay

# # %% ==========================================================================
# # BC pretraining
# # =============================================================================
# from expert_trajectories import transitions, agg_rew
# from imitation.algorithms import bc

# # Run pretraining
# bc_trainer = bc.BC(
#     observation_space=vec_env.observation_space,
#     action_space=vec_env.action_space,
#     demonstrations=transitions,
#     rng=rng,
# )
# bc_policy = bc_trainer.train(n_epochs=10)

# %% ==========================================================================
# Run algorithm
# =============================================================================
from imitation.algorithms.adversarial.gail import GAIL
from expert_trajectories import transitions, agg_rew
from stable_baselines3 import PPO

# # Ensure your transitions are properly formatted as list of dicts
# formatted_transitions = []
# for transition in transitions:
#     formatted_transitions.append({
#         # Flatten observations as before
#         'obs': np.array(list(transition['obs'].values()), dtype=np.float32),  # Flatten and cast to float32
#         # Ensure the action is wrapped in an array-like structure (e.g., np.array)
#         'acts': np.array([transition['acts']], dtype=np.float32),  # Wrap action in a numpy array
#         # Flatten next observations as well
#         'next_obs': np.array(list(transition['next_obs'].values()), dtype=np.float32),  # Flatten and cast
#         'dones': transition['dones']  # This can stay as a scalar (bool)
#     })

# Initialize generator
gen_algo = PPO("MlpPolicy", vec_env, verbose=1,
    # n_steps=2048,
    ent_coef=0.04,
    policy_kwargs=policy_kwargs,
    # learning_rate=custom_lr_schedule(1),  # Decreasing learning rate
    # learning_rate=linear_schedule(0.3),  # Decreasing learning rate
    learning_rate=3*10**-4,
    # normalize_advantage=True,
    gamma=0.97,
    batch_size=256,
    # tau=0.01,
    # replay_buffer_ratio=256,
    # replay_buffer_size=3*10**6
)
# # Load the BC pre-trained policy into the generator
# gen_algo.policy = bc_policy

# Initialize discriminator
GAIL_trainer = GAIL(
    demonstrations=transitions,  # Expert demonstrations
    venv=vec_env,
    demo_batch_size=64,
    gen_algo=gen_algo,
    reward_net=reward_net,
    # gamma=3*10**-5, # Also try 10**-6
    # replay_buffer_size=3*10**6

)

# Final training
GAIL_trainer.train(total_timesteps=20000)

# %% ==========================================================================
# Evaluate rewards and success rates
# =============================================================================
from stable_baselines3.common.evaluation import evaluate_policy

## Rewards
# Evaluate GAIL policy on the environment
reward_mean, reward_std = evaluate_policy(GAIL_trainer.policy, vec_env, n_eval_episodes=150)
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
    GAIL_trainer.policy, vec_env, n_eval_episodes=120,
    return_episode_rewards=False,  # We only need success info
    deterministic=True,
    warn=False,
    callback=success_callback)
# Compare GAIL to export
# Get success rates from expert_trajectories
emp_succ = [transitions.infos[i]['success'] for i in range(len(transitions))]
print(f"GAIL Model Success Rate: {sum(mod_succ):.2f}%")
print(f"empirical Success Rate: {sum(emp_succ)/27:.2f}%")

# %% ==========================================================================
# Evaluate actions distributions
# =============================================================================
# Get model actions and expert_trajectories
actions_GAIL = []  # Collect actions from the GAIL model
traject_GAIL = []  # Collect states from GAIL expert_trajectories
for _ in range(450):
    obs = vec_env.reset()
    for day in range(5):
        traject_GAIL = traject_GAIL + [str(list(obs[i])) for i in range(len(obs))]
        action, _ = GAIL_trainer.policy.predict(obs, deterministic=True)
        obs = vec_env.step(action)[0]
        actions_GAIL = actions_GAIL + [list(action[i])[0] for i in range(len(action))]
# Get expert actions and expert_trajectories
actions_expert = [transitions.acts[i][0] for i in range(len(transitions))]  # Collect actions from the expert
traject_expert = [str(list(transitions.obs[i])) for i in range(len(transitions.obs))]
pd_traj_ex = pd.DataFrame([traject_expert, actions_expert]).T
pd_traj_ex.columns = ['states', 'actions']
ag_traj_ex = pd_traj_ex.groupby(["states"]).mean()

# Aggregate state-actions for model
pd_traj_GAIL = pd.DataFrame([traject_GAIL, actions_GAIL]).T
pd_traj_GAIL.columns = ['states', 'actions']
ag_traj_GAIL = pd_traj_GAIL.groupby(["states"]).mean()
# Compare
trajs = pd.merge(ag_traj_ex, ag_traj_GAIL, left_index=True, right_index=True, how='left')
trajs.columns = ['expert', 'GAIL_model']
# Compare state-action pairs
correlation = trajs[['expert', 'GAIL_model']].corr()
print(correlation)

# # Plot action distributions
# import matplotlib.pyplot as plt
# plt.hist(actions_GAIL, bins=20, alpha=0.5, label="GAIL Policy")
# plt.hist(ag_traj_ex, bins=20, alpha=0.5, label="Expert Policy")
# plt.legend()
# plt.xlabel("Actions")
# plt.ylabel("Frequency")
# plt.title("Action Distributions: GAIL vs Expert")
# plt.show()
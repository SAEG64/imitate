#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:34:31 2025

@author: sergej
"""

import pandas as pd
import os

# Parse data
path = os.path.dirname(__file__)+"/"
os.chdir(path)
data = pd.read_csv("data_beh/datall_cat.csv")
# Get first unique lines
data_uniqs = data.groupby('x4_index_forests', as_index=False).first()

# =============================================================================
# Get initial energy points
# =============================================================================
# data_uniqs.columns[:7]
LP_init = list(pd.unique(data_uniqs['x6_continuous_energy_trial_start']))

# =============================================================================
# Get forests
# =============================================================================
# data.columns[56:]
forests = data_uniqs[['x57_weather_1_gain_magnitude','x58_weather_2_gain_magnitude','x59_weather_1_p_gain','x60_weather_2_p_gain']]
# Get forest parameters (gains and psrobabilities)
gains = forests[forests.columns[:2]].to_dict(orient='list')
gains = {i: gains[key] for i, key in enumerate(gains.keys())}
psucc = forests[forests.columns[2:]].to_dict(orient='list')
psucc = {i: psucc[key] for i, key in enumerate(psucc.keys())}

# # Evaluate data
# agg_dat = data.groupby('x4_index_forests', as_index=False).mean()
# dat_cut = agg_dat[['x6_continuous_energy_trial_start','x57_weather_1_gain_magnitude','x58_weather_2_gain_magnitude','x59_weather_1_p_gain','x60_weather_2_p_gain']]

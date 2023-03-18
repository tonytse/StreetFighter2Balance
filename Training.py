#!/usr/bin/env python
# coding: utf-8

import StreetFighter as sf

get_ipython().system('python3 -m retro.import ./roms # Run this from the roms folder, or where you have your game roms')


import numpy as np
import math
import csv
import json

# stable baseline
from sb3_contrib import TRPO

model_params = None
with open('TRPO.json') as f:   
    model_params = json.load(f)

kLearnTimesteps = 50_000
kIteration = 5


# Loop for each characters
for c in sf.kCharacters:
    logDir = f'./log_{c}/'
    model = None
    
    # Loop for n iteration
    for i in range(kIteration):

        # vs other character
        for opponent in sf.kCharacters:
            if c == opponent:
                continue

            # Set the state name
            state = f'L4_{c}_{opponent}'
            
            # create environment
            env = sf.CreateEnv( state, 1, logDir, 6 )
            
            # create model
            if model is None:
                model = TRPO("MlpPolicy", env, verbose=0, tensorboard_log=logDir, tb_log_name=f'{c}_{opponent}, **model_params )

            # set environment
            model.set_env(env)
            model.learn(total_timesteps=kLearnTimesteps)
            env.close()
    # save the model
    model.save( f'model/{c}.zip')
    print( f'Done -> {c}')    





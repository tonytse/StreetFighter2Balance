#!/usr/bin/env python
# coding: utf-8
import StreetFighter as sf
import os
os.system('python3 -m retro.import ./roms # Run this from the roms folder, or where you have your game roms')

import optuna
import json
# stable baseline
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import RecurrentPPO, TRPO, QRDQN
from stable_baselines3.common.evaluation import evaluate_policy


kLearnTimesteps = 150_000


def optimize_trpo_agent(trial):
    model_params =  {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512]),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048, 4096, 8192]),
        'gamma': trial.suggest_float('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, .99),
        "n_critic_updates": trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30]),
        "cg_max_steps": trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30]),
        "target_kl": trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
        
    }
    env = sf.CreateEnv( 'L4_Ryu_Guile', 1, None, 6 )
    model = TRPO("MlpPolicy", env, verbose=1, **model_params )
    model.learn(total_timesteps=kLearnTimesteps)
    mean_reward, _ = evaluate_policy(model, env)
    env.close()
    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_trpo_agent, n_trials=20)
study.best_params



with open('TRPO.json', 'w') as outfile:
    json.dump(study.best_params, outfile)

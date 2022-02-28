import gym
import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gym_petrinet.envs import render
from gym_petrinet.envs.petri_env import PetriEnv


#%% load , make , chek custom petrinet env 

sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")

env = PetriEnv()
check_env(env, warn=True)
#env = make_vec_env(lambda: env, n_envs=1)

#%% Train model 

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e6))
# Save the agent
model.save("Trained models\dqn_model3_dot3.zip")
#del model  # delete trained model to demonstrate loading

#%% Test model 

episodes=100
model = DQN.load("Trained models\dqn_model3_dot3.zip")

for ep in range (episodes):
    
    dones =False
    ep_reward=0
    obs = env.reset()
    
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action,False,ep) 
        ep_reward+=rewards

    print(ep_reward)
    
render()
    
#%%Ntesting 


import gym
import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

#%% load , make , chek custom petrinet env 

sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from petri_env import PetriEnv
env = PetriEnv()
check_env(env, warn=True)
#env = make_vec_env(lambda: env, n_envs=1)

#%% Test saved model 

episodes=1
#model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
model = DQN.load("Trained models\dqn_model2_dot.zip")


for ep in range (episodes):
    
    dones =False
    ep_reward=0
    ep_steps=0
    obs = env.reset()
    
    while not dones:
        action, _states = model.predict(obs,deterministic=True)
   
        obs, rewards, dones, info = env.step(action,1,ep) 
        ep_reward+=rewards
        ep_steps+=1
        #if ep_steps > 100 : break 

    print("Total episode Reward : {}".format(ep_reward))
    print("Total episode Steps : {}".format(ep_steps))
   
env.render(replay=False,continues=True)

#%%

#%%



 
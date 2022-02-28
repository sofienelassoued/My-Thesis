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
#%% Train the model (for the Uni Pc )

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(3e5))
# Save the agent
model.save("Trained models\dqn_model3_dot2.zip")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward,std_reward)




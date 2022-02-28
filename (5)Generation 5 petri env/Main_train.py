
import sys
import os
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from environement import Environement

env=Environement()
#nv = make_vec_env(lambda: env, n_envs=1)
#check_env(env, warn=True)



#%% Train the model (for the Uni Pc )
# Instantiate the agent

env.reset()

model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(3e5))
# Save the agent
model.save("Trained models\dqn_model6.5.zip")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward,std_reward)

#%%


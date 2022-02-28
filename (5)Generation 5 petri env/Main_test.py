
import sys
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from environement import Environement
import matplotlib.pyplot as plt


env=Environement()
check_env(env, warn=True)



#%% Test saved model 

episodes=1

#model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
model = DQN.load("Trained models\dqn_model6.3.zip")


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


#%%plot Dater and counter 

x = list(range(env.simulation_clock))
# corresponding y axis values
y = env.petri.Places_obj[1].place_dater
  
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('Steps')
# naming the y axis
plt.ylabel('number of token ')
  
# giving a title to my graph
plt.title('dater')
  
# function to show the plot
plt.show()
#%%

print (env.petri.Transition_obj[1].transition_history)

 
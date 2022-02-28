# -*- coding: utf-8 -*-

import sys
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from environement import Environement
import matplotlib.pyplot as plt

env=Environement()


#%% testing 

action=9
marking=env.petri.get_marking()
possible,inprocess=env.petri.enabled(action,marking)

print(possible)


fire=env.petri.fire_transition (action,marking,possible,inprocess)

print(fire)


#%%

     


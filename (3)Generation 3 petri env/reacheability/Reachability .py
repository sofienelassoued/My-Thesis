import gym
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from petri_env import PetriEnv
env = PetriEnv()

#%%

M0=np.array(env.initial_marking).tolist()

     
(V,E,v0)=([M0],[],M0) 
      
work=[M0]
      
while len (work)>0:

    for i in work : 
         work.remove(i)
   
         df =env.possible_firing(i) 
         enabled =df.loc[df["Firing enabled"] ==True].index.tolist()
         i_enabled=[env.Transition_names.index(i) for i in enabled]
   
         for t in i_enabled:
             Nxmarking,features,fired,inprocess=env.fire_transition (t,i)
             NewM=np.array(Nxmarking).tolist()
        
             if NewM not in V :
                V.append(NewM)
                work.append(NewM)
                E.append((i,t,NewM))
           
    
print(E[0])
      




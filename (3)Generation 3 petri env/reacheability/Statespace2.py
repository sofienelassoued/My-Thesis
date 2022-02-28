import torch
import torch.nn as nn
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from petri_env import PetriEnv
env = PetriEnv()

#%%

def Reachability ():
    
    M0=np.array(env.initial_marking).tolist()

    Xreach=[]
    U=[M0]
    
    
    while len (U) >0 :  
        
        
        
        for i in U :         
            #choose a state i in U and move it to Xreach;

            Xreach.append(i)    #Xreach contains the known states already explored
            U.remove(i)         #U contains the known states not yet explored
            
            
           # calculate J the set of all possible next sates 
            Nset=[]
            df =env.possible_firing(i) 
            enabled =df.loc[df["Firing enabled"] ==True].index.tolist()
            i_enabled=[env.Transition_names.index(i) for i in enabled]          
            
            for t in i_enabled:
                Nxmarking,features,fired,inprocess=env.fire_transition (t,i)
                Nset.append(np.array(Nxmarking).tolist())
                
            #for each j ∈ N (i) do
                
            for j in Nset:       
                if (j not in Xreach ) and (j not in U) :
                   U.append(j)

    return(Xreach)
                

Xreach= Reachability ()


print((Xreach))
print(len(Xreach))


#%%


            
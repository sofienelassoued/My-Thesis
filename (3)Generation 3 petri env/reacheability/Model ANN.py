
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

     
    (V,E,v0)=([M0],[],M0) 
      
    work=[M0]
    iteration=0
      
    while len (work)>0:
        
        print ("Itheration Number :{}" .format (iteration))
        print (work)
        
        
        
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
                      E.append((i+[t],NewM))
                      
                      
        iteration+=1
                      
       
                      
      
                      
          
                      
                      
    df = pd.DataFrame.from_records(E, columns =['features', 'Label'])
    train = torch.tensor(df['features'], dtype=torch.int64)
    labels=cats = torch.tensor(df['Label'], dtype=torch.int64)
                 
    return (df,train,labels)

df,train,labels = Reachability ()


#print(df)


#%%



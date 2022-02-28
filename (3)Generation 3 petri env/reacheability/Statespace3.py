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
    
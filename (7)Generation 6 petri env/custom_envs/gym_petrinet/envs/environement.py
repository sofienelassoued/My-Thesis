#%% Libraries importation
import gym

import torch
import copy
import numpy as np
from gym import Env
from gym import spaces
from petrinet import Petrinet
from render import snapshot_creator,Tree_creator,display


#%% Main environement 

class Environement(gym.Env):

    def __init__(self,ModelID=9):
      
        super().__init__() 
        
        self.petri=Petrinet(ModelID)
        self.petri.load_model()
        
        self.max_steps=300
        self.Terminal=False
        self.simulation_clock=0
     
             
        self.goal=10
        self.episode_reward=0
        self.explored_states=[]    
        self.grafic_container=[]
        self.saved_render=copy.deepcopy(self.grafic_container)

        self.action_space = spaces.Discrete(self.petri.NTRANSITIONS)      
        self.observation_space = spaces.Box(low=-100, high=100, shape=(self.petri.NPLACES,),dtype=np.int32) 
           
        '''
  
  Description:
      
  Observation:
      
  Actions:
      
      Type: Discrete (Number of available transitions U idle )
      
  Reward:
      
  Starting State:
      
  Episode Termination:
      

'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    def step(self, action,testing=False,episode=0):
        
        reward=0
        done=False
        info = {}
        observation=[]
        self.simulation_clock+=1
        current_marking=self.petri.get_marking()
        
  
        #---Firing step-------------------
        
        possible,inprocess=self.petri.enabled(action,current_marking)
        Next_marking,fired=self.petri.fire_transition (action,current_marking,possible,inprocess)
          
        #---Test termination  --------------
        
        self.Terminal=all([self.petri.enabled(i,current_marking)[0] == False for i in range(self.petri.NTRANSITIONS)]) #no enabled transition
        if self.simulation_clock> self.max_steps:self.Terminal=True #max steptes exceeded
      
  
        observation=Next_marking     
        reward,firing_info=self.reward(Next_marking,fired) 
        info.update({"Action": self.petri.Transition_names[action]})
        done=self.Terminal
        

        self.episode_reward+=reward
        

        #---Creat a snapshop  ------------
        
        if testing==True: # take a snapshot in testing false         
           graph=graph_generater(self.petri.Places_obj,self.petri.Transition_names,self.petri.Transition_names[action],fired,inprocess)
           self.grafic_container.append(snapshot_creator(graph,self.simulation_clock,reward,self.episode_reward,firing_info,episode=0))
           print ("Screenshot for step {} created ".format(self.simulation_clock))
           
           
        #---Genrate dater and counter and update process --------
        
        for i in self.petri.Places_obj :
            i.place_dater.append(len(i.token_list))
            i.process_time-=1
            
           
        for i in self.petri.Transition_obj:
            if i.name==str ("T"+str(action)) and possible :
                i.transition_history.append(self.simulation_clock)
  
        #print(f"action {action}\n observation {observation}\n reward {reward}\n info {firing_info}\n" )
  
        return observation, reward, done, info
        

    def reward(self,Next_state,fired): 
        
        reward=0 
        IOB=14
                    
        if  int(self.petri.marking[IOB])>self.goal:
            
            # Goal achieved  
            reward=+10
            firing_info="Goal achieved !! " 
            self.Terminal=True  

              
        elif self.Terminal==True :    
            # dead lock
            reward=-10
            firing_info="Dead lock"
    
        elif fired == False :
            # firing halted
            reward=-2
            firing_info="firing halted" 
                
        else :
            # firing sccessful              
            reward=-1
            firing_info="firing successful"
        
        
        self.petri.delivered1=int(self.petri.marking[IOB]) 

        
       # print(reward ,firing_info)
        
        return reward ,firing_info
          
 
        
    def reset(self):
        
                    
        self.Terminal=False  
        self.episode_reward=0
        self.simulation_clock=0
        self.grafic_container=[]
        self.episode_actions_history=[] 
        self.petri.reset_places()
    
        array=np.array(np.zeros((self.petri.NPLACES,),dtype=np.int32))
        
        return array
      

    def render(self,replay=False,continues=True):
        display(200,self.grafic_container,self.saved_render,replay,continues)

    
    def close():
        pass
  

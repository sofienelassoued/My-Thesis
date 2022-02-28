
import os
import sys
import math
import copy
import pygame 
import random
import itertools
import statistics
import graphviz
from graphviz import Digraph
from Tropical import next_states


sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")
from petrinet import  Petrinet
from environement import Environement
from render import Tree_creator






env=Petrinet(11)
env.load_model()




#%% Monte carlo search tree 


#---------------------------Max Plus---------------------------------  
# [ 0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11	,12,  13, 14, 15]]
# [T1,T2,T3,T4,T5,T6,T7,T8,  U11,U12,U21,U22,U31,U32,UO2,UO1]]


# ------------------------Petrinet--[11]---------------------------- 

# [	T1	T2	T3	T4	T5	T6	T7	T8	U11	U12	U21	U22	U31	U32	UO2	UO1]
# [IB1	IB2	OB	P11	P12	P21	P22	P31	P32	S11	S12	S21	S22	S31	S32	OB2	OB1]


# -----------------------Correspondance -----------------------

#[T1:U11 , T2:U21 , T3:U31 , T4:UO1 , T5:U22 , T6:U32 , T7:U12 , T8:UO2]
#[0:8, 1:10 , 2:12 , 3:15 , 4:11 , 5:13 , 6:9 , 7:14]

Nodes=[]

    
class Node():
    
    ID = itertools.count()
    def __init__(self, state):
        self.id = next(self.ID)
        self.state = state
        self.visit_count = 0
        self.value_sum = 0 
        self.Action =0  # the action origin of the node creation
        
        
        self.path = []
        self.parent=[]
        self.children = []
 
        self.selected=False 
        
        
    def __str__(self):
              return(f"State {self.state},Marking {self.marking} ,visits {self.visit_count},Cost {self.value_sum},ID {self.id}" )
              
def EXPANSION (node):
    
    node.selected=True
    possible_states=next_states(node.state)  
    
    
    for i in  range (len (possible_states)): 
        
        add_Node(possible_states[i])   
        possible,marking=env.enabled(i,node.marking)
        
        Nodes[-1].path.extend(node.path)
        Nodes[-1].path.extend([node.id])
        Nodes[-1].parent=node
        Nodes[-1].marking=marking
        
        node.children.append(Nodes[-1])

        



def UCB1 (node,exploration_rate): 
    
    parent_visites=0
    if node.visit_count==0 :
        return -1*math.inf
    
    else :

        parent_ID=node.path[-1]
        for i in Nodes :
            if i.id==parent_ID :  
                parent_visites=i.visit_count

        UCB_epr=(node.value_sum/node.visit_count )
        UCB_epl=exploration_rate*(math.sqrt(math.log(parent_visites)/node.visit_count))
       
        UCB=UCB_epr+UCB_epl
        
       # print(f'exploration{UCB_epl} , exploitation {UCB_epr}')
 
    return UCB


def SELECTION(node,exploration_rate=10):
    
     UCB=[]
  
     for i in node.children : 
         UCB.append(UCB1 (i,exploration_rate))
 
     min_ucb = min(UCB)
     min_index = UCB.index(min_ucb)
     
 

     return(node.children[min_index],min_index)



 

def SIMULATION ( leaf,simepoch=1,Maxsteps=1000 ) :
    
    samples=[]
    
    state=copy.copy (leaf.state)  
    i=0
    j=0
    
    for i in range (simepoch):
        
        while i < Maxsteps :
            possible_states =next_states(state)  
            randomstate= random.choice(possible_states) 
            #obejctive is to fire the X4 and  X8 10 times 
            # the transition is fires if  the dater is diffrent    
    
            if randomstate[7]!= leaf.state[7]: j+= 1
            if j>10 :break
    
            state=randomstate    
            i+=1   
            
        samples.append (state[7])   
        cost=statistics.mean(samples)
        
    return  cost

  

def BACKPROPAGATION (current_node,cost):
    
    
    path=current_node.path  
    current_node.value_sum+=cost
    current_node.visit_count+= 1
    
    for i in path : 
        Node_update( i , cost )
        

    
def find_Node(ID): #findes the node index in the Node list based on ID 

     for i in range (len(Nodes)):
        if Nodes[i].id==ID : return (i)
        

def Node_update( ID, cost ):
    for i in Nodes :
        if i.id==ID :     
            i.visit_count+=1
            i.value_sum+=cost
            

def add_Node (state): # check if node existes and add 
    
    exist=False
    
    for i in Nodes :
        if i.state==state : exist=True
        
    if exist==False :  
        Nodes.append (Node(state))
        
    else :Nodes.append (Node(state)) # comment to delete duplicates
    
    
def initiaion ():
    
    initial_state=[-1,-1,-1,-1,-1,-1,-1,-1]
    Nodes.append (Node(initial_state))
    Nodes[0].path=[]
    Nodes[0].marking=env.marking
    
    EXPANSION (Nodes[0])
    


def Action_filter():
    
    
    marking=env.marking
    possible_actions=[]
    
    for i in range (0,16) : 
        
        possible,next_marking=env.enabled(i,marking)
        
        if possible : 
            possible_actions.append(i)
            
        
    return  possible_actions


     

#%%
   
testing =1

if testing==1:
    
    initiaion ()
    current_node=Nodes[0]
    
    Grafic_container=[]
    saved_render=[]
    
    
    for i in range (100):
  
         #exploration_rate=100*math.exp(-0.01*i)
         exploration_rate=0.001
         
         print(f"-----evaluation Number {i} Bias {exploration_rate}----------")
    
          
         if len(current_node.children)==0:
             
             if current_node.visit_count==0:
        
                 mcts_cost =SIMULATION (current_node,100)
                 BACKPROPAGATION (current_node,mcts_cost)
                 current_node=Nodes[find_Node(current_node.parent.id)]
              
             else:
                 
                 EXPANSION (current_node) 
                 Grafic_container.append(Tree_creator(Nodes,i))
                 current_node,action = SELECTION(current_node.parent,exploration_rate)
                 cost =SIMULATION (current_node,100)
                 BACKPROPAGATION (current_node,cost)
                 current_node=Nodes[find_Node(current_node.parent.id)]
    
         else : 
     
             current_node,action = SELECTION(current_node,exploration_rate)


     


#%%


print (env.enabledMax(env.marking))


    


import os
import sys
import math
import copy
import pygame 
import random
import itertools
import statistics
import graphviz


sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")


from graphviz import Digraph
from Tropical import next_states
from petrinet import  Petrinet
from environement import Environement
from render import snapshot_creator,Tree_creator,display





env=Petrinet(9)
env.load_model()




#%% MCST

# [T0,T1,T2,T3,T4,T5,T6,T7,  U1,U11,U21,U31 ,  U2,U22,U32,U12]

# 0 T1      # 8 U1
# 1 T2      # 9 U11
# 2 T3      # 10 U12
# 3 T4      # 11 U2
# 4 T5      # 12 U21
# 5 T6      # 13 u22
# 6 T7      # 14 U31
# 7 T8      # 15 U32



Nodes=[]


     
class Node():
    
    ID = itertools.count()
    def __init__(self, state):
        self.id = next(self.ID)
        self.state = state
        self.visit_count = 0
        self.value_sum = 0 
        self.marking=[]
        
        self.path = []
        self.parent=[]
        self.children = []
 
        self.selected=False 
        
        
    def __str__(self):
              return(f"State {self.state},Marking {self.marking} ,visits {self.visit_count},Cost {self.value_sum},ID {self.id}" )
              
def EXPANSION (node):
    
    node.selected=True
    mask=Action_filter()
    
  
    possible_states=next_states(node.state,mask)  
    
    
    for i in  range (len (possible_states)): 
        
        add_Node(possible_states[i])  
        possible,marking=env.enabled(mask[i],node.marking)
        
        Nodes[-1].path.extend(node.path)
        Nodes[-1].path.extend([node.id])
        Nodes[-1].parent=node
        Nodes[-1].marking=marking
        
   
        node.children.append(Nodes[-1])
        node
        



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
    


def Action_sequence(Nodes):
    
    Actions =[]
    Sequence=[]
 
    
    for n in range(1, len (Nodes)) :# find the expanded nodes id     
        if Nodes[n].selected==True :
            Actions.append (Nodes[n])
       
    for a in Actions:     
        for j in range(len(a.parent.children)):
            
            if a.id==a.parent.children[j].id:
                Sequence.append(j)
                
    return(Sequence)


def Petri_update(action):
    
    #conver_dict= {0:8,1:9,2:12,3:14,4:11,5:13,6:15,7:10 }#from MCTS to Petri
    #action=conver_dict[action]     
    
    
    marking=env.marking
    possible=env.enabled(action,marking)
    new_state,firing_confirmation=env.fire_transition (action,marking,possible)
    
    
    return new_state
    


def Petri_supervisor(cost,action):
    
    
    # 0 T1      # 8 U1
    # 1 T2      # 9 U11
    # 2 T3      # 10 U12
    # 3 T4      # 11 U2
    # 4 T5      # 12 U21
    # 5 T6      # 13 u22
    # 6 T7      # 14 U31
    # 7 T8      # 15 U32

    marking=env.marking
    possible,marking=env.enabled(action,marking)
    if not possible : cost+=1000

    return (cost)


def Action_filter():
    
    
    marking=env.marking
    possible_actions=[0,4]
    
    for i in range (8) : 
        
        possible,next_marking=env.enabled(i,marking)
        
        if possible : 
            possible_actions.append(i)
        
        
    return  possible_actions


def Tree_creator_V2(Nodes,simulation_clock=0):
    
           
    graph =  graphviz.Digraph('structs', filename='step',format='jpg' ,
                         node_attr={'shape': 'record'})
  
    for node in Nodes:  
        

        combination=str(str(node.value_sum)+'|{<f0>'+str( "ID:"+ str (node.id)) +'|'+ str(node.state)+'|<f1>'+ str(node.marking)+'}|'+str(node.visit_count))
        
        graph.node(str(node.id), combination)
        
        
        
    graph.edge('0:f1','1:f0') 
    graph.edge('0:f1','2:f0') 
        

        
        
          
    graph.view()

            

#%%
   
testing =1

if testing==1:
    
    initiaion ()
    current_node=Nodes[0]
    
    Grafic_container=[]
    saved_render=[]
    
    
    for i in range (50):
  
         exploration_rate=100*math.exp(-0.01*i)
         #exploration_rate=0.001
         
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
                 #cost=Petri_supervisor(cost,action)
                 BACKPROPAGATION (current_node,cost)
                 current_node=Nodes[find_Node(current_node.parent.id)]
    
         else : 
     
             current_node,action = SELECTION(current_node,exploration_rate)

    #display(200,Grafic_container,saved_render,continues=False)
        


#%%


Tree_creator_V2(Nodes,simulation_clock=0)




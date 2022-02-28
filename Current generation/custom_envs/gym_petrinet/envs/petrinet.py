# %% libreries 

import os
import copy 
import numpy as np
import pandas as pd


# %% Main code

class Petrinet():
     
    def __init__(self,modelID):

        self.modelID=modelID
        self.path = os.getcwd()+"\modelisation/"+str(self.modelID)+".html"
        self.Forwards_incidence = pd.read_html(self.path,header=0,index_col=0)[1]
        self.Backwards_incidence= pd.read_html(self.path,header=0,index_col=0)[3]
        self.Combined_incidence = pd.read_html(self.path,header=0,index_col=0)[5]
        self.initial_marking =pd.read_html(self.path,header=0,index_col=0)[9].loc["Current"]
        
        
        self.marking= np.array(tuple(self.initial_marking),dtype=np.int32)        
        #self.processing_time = {"S11":1,"S21":2,"S31":3,"S12":3,"S32":2,"S22":1}
        self.processing_time = {"S11":0,"S21":0,"S31":0,"S12":0,"S32":1,"S22":1}
              
        self.Places_names= self.Forwards_incidence.index.tolist()
        self.NPLACES=len( self.Places_names)
        self.Places_obj=[]

            
        self.Transition_names= self.Forwards_incidence.columns.tolist()
        
        self.NTRANSITIONS=len( self.Transition_names)
        self.Transition_obj=[]
        
        self.delivered1=self.initial_marking["OB"]

        
    class token: 
        def __init__(self,ID=0,color="N"):
            
            self.ID =ID
            self.color=color             
            self.token_history=[]
            
        def __str__(self):
            return(f" ID:{self.ID}, color {self.color } " )
                  

    class Place: 
        
        def __init__(self,name,token_list,In_arcs,Out_arcs,):
            

              self.name =name
              self.token_list=token_list  
              self.In_arcs=In_arcs
              self.Out_arcs=Out_arcs 
              

              
              self.place_dater=[]
              self.process_time=0
              
              

        def __str__(self):          
            return(f"Place name {self.name}  Tokens: {len(self.token_list)} Input:{self.In_arcs}  Output{self.Out_arcs } " )
                   
    
    class Transition:
          def __init__(self,name,In_arcs,Out_arcs,index):
              
              self.name =name
              self.index=index
              self.In_arcs=In_arcs
              self.Out_arcs=Out_arcs
              self.transition_history=[] 
              self.input_transition=False
              
                   
       
          def __str__(self):
              return(f"Tansition name {self.name}   Input:{self.In_arcs}  Output{self.Out_arcs }" )
          

            
    def load_model(self):
        

        #--- Loop for every place -------------------------------
        for i in self.Places_names:          
          
      
          name=i
          In_arcs=[]
          Out_arcs=[] 
          token_list=[]
 
             
          for j in self.Forwards_incidence.columns.tolist() :   
              if self.Forwards_incidence.loc[i,j]==1:
                  In_arcs.append(j)
            
          for k in self.Backwards_incidence.columns.tolist() :   
              if self.Backwards_incidence.loc[i,k]==1:
                  Out_arcs.append(k)  
                  
          for l in range (int(self.initial_marking[i])):
              token_list.append (self.token()) 
              
          
          self.Places_obj.append(self.Place(name,token_list,In_arcs, Out_arcs))
    
        #--- Loop for every transition --------------------------
        
        index=0;
        
        for i in self.Transition_names:  

            In_arcs=[]
            Out_arcs=[]         
            name=i
                
            for j in self.Forwards_incidence.index.tolist() : 
                if self.Forwards_incidence.loc[j,i]==1:
                    Out_arcs.append(j)
            
            for k in self.Backwards_incidence.index.tolist() : 
                if self.Backwards_incidence.loc[k,i]==1:
                    In_arcs.append(k)
                      
            self.Transition_obj.append(self.Transition(name,In_arcs, Out_arcs,index))
            index +=1
            
            
            
        for i in self.Transition_obj: # check if its aninput transition 
            if len (i.In_arcs)==0 :
                i.input_transition=True
                #print("input transition ")

        
        print(f"file loaded successfully from {self.path} \n Number of places :{self.NPLACES} \n Number of transitions :{self.NTRANSITIONS}")
        
        
    def plus_token(self,place,tokens_number=1,color="N"):   
        
        token_list=[]
        
        for i in range (tokens_number): 
            token_list.append (self.token())
            
        for m in token_list:
              m.ID=id(m)

        for i in self.Places_obj :
            if i.name==place :
                i.token_list.append(token_list)

        
    def minus_token (self,place,tokens_number=1):   
        

        for i in self.Places_obj :
            if i.name==place :                
                for j in range (tokens_number) :
                    i.token_list.pop()                      

    
    def get_marking(self):  
        
        marking=[]
        for i in self.Places_obj:
            marking.append((len(i.token_list)))        
        return(np.array(marking,dtype=np.int32))
    
        
    def reset_places(self) :
        
        for i in self.Places_obj :
            i.token_list=[] 
            i.place_dater=[]
            
        self.marking=self.get_marking()
            

                
                
    def load_state(self,marking):
        
        self.reset_places()
             
        for i in  range (len(self.Places_obj)) :
          
            for j in range ( marking[i]) :
                self.Places_obj[i].token_list.append(self.token())
                
        self.marking=self.get_marking()
                

        
    def enabled(self,action,marking):
        
    
        possible =False

        firing_array =np.zeros(self.NTRANSITIONS,dtype=np.int32)
        firing_array[action]=1   
        Next_marking=(firing_array.dot(self.Combined_incidence.T)+ marking) 
        
        
        if  len( self.Transition_obj[action].In_arcs )==0 :  
            possible=True # input transition 

        else : possible = all([Next_marking[i] >= 0 for i in range (len( Next_marking))])
                              
        return possible,Next_marking
    
    
    
    def enabledMax(self,marking):
        
        #correspondance = {0:8, 1:10 , 2:12 , 3:15 , 4:11 , 5:13 , 6:9 , 7:14}
        
        
    
        
        tran_state=[]
        input_transitions =[];
        process_transitions =[];
        original_marking=copy.deepcopy(marking)
         
        
        # fire all inputs and then check actually wich process transition are enabled then restore the original marking 
        
        for T in self.Transition_obj:# devide tramsitions into input and process 
            if T.input_transition==True :input_transitions.append(T)
            else :process_transitions.append(T)
            
            
            
        for Tin in  input_transitions : # fire all input transition
           marking,delivered=self.fire_transition (Tin.index,marking,True)
            
            
        for Tpro in process_transitions : 
            
            possible,Next_marking=self.enabled(Tpro.index,marking )
            if  possible :
                tran_state.append (Tpro.name)
           
   
        self.load_state(original_marking) # restore the original marking
        
        
     
        
        return (tran_state)
            
  
    
    
    def fire_transition (self,action,marking,possible):


      #---if not enabled ---------------

      if  not possible  :
         
          #print("firing Halted! no tokens ")
          return (self.get_marking(),False) 

      #---if firing successful---------------
   
        
      if possible:
          
            transition=None 
            for i in self.Transition_obj :  #find the transition object        
                if self.Transition_names[action] ==i.name:
                    transition=i                 
            
            for j in  transition.In_arcs :
                self.minus_token (j)
                
            for k in  transition.Out_arcs :
                self.plus_token (k)          
                         
            #print("fired successfuly ")
            self.marking=self.get_marking()
            return (self.get_marking(),True) 
        
        
        
#%%
            
            
          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
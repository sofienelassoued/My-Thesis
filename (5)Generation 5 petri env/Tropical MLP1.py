import random
from csv import reader
import numpy as np
import copy
import itertools
from itertools import product 
from sklearn import preprocessing
import pandas as pd



#%% Max Plus

E=-1
e=0 

a="t11"
b="t12"
c="t13"
x="t22"
z="t23"
w="t21"


a=1
b=2
c=3
x=1
z=2
w=3


A = [[E,E,E,E,E,E,E,E],
    [E,a,E,E,E,E,E,E],
    [E,E,b,E,E,E,E,E],
    [E,E,E,c,E,E,E,E],
    [E,E,E,E,E,E,E,E],
    [E,E,E,E,x,E,E,E],
    [E,E,E,E,E,z,E,E],
    [E,E,E,E,E,E,w,E]]

B = [[0,0,E,E,E,E,E,E],
    [E,E,0,E,E,E,E,E],
    [E,E,E,0,E,E,E,E],
    [E,E,E,E,E,E,E,E],
    [E,E,E,E,0,E,0,E],
    [E,E,E,E,E,E,E,0],
    [E,E,E,E,E,0,E,E],
    [E,E,E,E,E,E,E,E]]

As =[[0,E,E,E,E,E,E,E],
    [E,E,E,E,E,E,E,E],
    [E,E,E,E,E,E,E,E],
    [E,E,E,E,E,E,E,E],
    [E,E,E,E,0,E,E,E],
    [E,E,E,E,1,0,E,E],
    [E,E,E,E,3,2,0,E],
    [E,E,E,E,6,5,3,0]]


X=[E,E,E,E,E,E,E,E]
U=[E,E,E,E,E,E,E,E]


hA =[[E,E,E,E],
     [a,E,E,E],
     [E,b,E,E],
     [E,E,c,E]]

hA1 =[[E,1,E,E],
     [E,E,1,E],
     [E,E,E,1],
     [E,E,E,E]]


hB =[[1,1,E,E],
     [E,E,1,E],
     [E,E,E,1],
     [E,E,E,E]]

hAs =[[0,E,E,E],
     [1,0,E,E],
     [1,2,0,E],
     [6,5,3,0]]


hU=[E,E,E,E]
hX=[0,0,0,0]



def Max(a,b):
    
    if a == E : return b
    elif b == E: return a
    
    try :return max  (a,b )
    except :return ("Max (" + str(a) +","+ str(b) +")" )
     
    

def Plus(a,b):
    if a == E : return a
    elif b == E : return b
    
    try :return int  (a+b )
    except :return (str(a) +"+"+ str(b)  )
    

def Matmul(A,B)  :
    
    result=[]
    
    if isinstance(B[0], list) :
    
        for i in range (len (A) ):
            result.append([E]* len(A[0]))
     
        # iterate through rows oE X
        for i in range(len(A)):
            
            # iterate through columns oE Y
            for j in range(len(B[0])):
               # iterate through rows oE Y
                for k in range(len(B)):
                    
                    result[i][j]=Max( result[i][j] , Plus (A[i][k] , B[k][j]))
                    
    else : 
            #print("its a vector multiplication")
            for i in range (len(B)):
                result.append(E)

            # iterate through rows oE X
            for i in range(len(A)):
            
               # iterate through rows oE Y
                   for k in range(len(B)):
                    
                    result[i]=Max( result[i] , Plus (A[i][k] , B[k]))
                      
    return(result)


def Matpower(A,n): 
    result=A
    for i in range (1,n):
        result=Matmul(A,result)
        
        
    return(result)


def Vecsum (A,B) :
    res=[]
    for i in  range (len (A)):
        res.append(Max(A[i],B[i]))       
    return res




def dataset_creator_states(n) : 
    
    dataset=[] 
    label=[]
    delta_label=[]
    
    pocket=[E,0]+list(range(1,n+1))
    
    
    states=list(itertools.product(pocket, repeat = 4))

    A_0= Matmul(hAs,hA1)
  
    for i in range (len(states)) :    
        
        AX=[]
        delta=[]
        AX=Matmul(A_0,states[i])
        label.append(AX)
      
        for j in range( len(AX)):
            delta.append(AX[j]-states[i][j])

        delta_label.append(delta)
        
    
    for k in range (len(states)) :
        dataset.append((list (states[k]),label[k],delta_label[k]))
        
    
    return(dataset)

def clean (weights):
    
    cweights=copy.deepcopy(weights)
    for i in range(len( cweights)):
                for j in range (len( cweights[0])):
                   if cweights[i][j]<-1 :cweights[i][j]=-1
                   else:cweights[i][j]=round(cweights[i][j],0)
    
    return (cweights)
        


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset



# Initialize a network
def initialize_layer(n_inputs, n_outputs,seedin=101):
    
    random.seed(seedin)
    weights=[]  

    for i in range (n_inputs):
       #weights.append([round(random.random(),2) for i in range(n_outputs)] )
        weights.append([E for i in range(n_outputs)] )
    return weights


# Calculate neuron activation for an input
def activate(neuron, weights, inputs):
     
    candidate=[]
    for i in range(len(inputs)):
        candidate.append (weights[neuron][i] + inputs[i])
           
        
    return (max(candidate), candidate.index(max(candidate)))

    
 
# Transfer neuron activation
def transfer(activation):
    
    if activation>=0 : return activation 
    else :return E

# Forward propagate input to a network output
def forward_propagate(weights, inputs):
    
    
    n_input=len(weights)
    
    layer_history=[]
    layer_out=[]
    
    for neuron  in range (n_input):         
        zactivation,position = activate(neuron,weights, inputs)
        neuron_out = transfer(zactivation)
        layer_out.append(neuron_out)
        layer_history.append(position)
  
    return layer_out, layer_history



def error (prediction,label):
    delta=[]
    for i in range (len(prediction)) : 
        delta.append((prediction[i]-label[i]))
      
    return delta
        
        

def back_propagate( weights,error,history,lr=0.1):
    

    for neuron  in range (len(weights)):   

       weights[neuron][history[neuron]]-=error[neuron]*lr

    return weights



#print(weights)

def training(dataset,weights,ep):
    
    data_lenth=len(dataset)

    
    for i in  range (data_lenth):        
        if i % 100==0 :
            print(f"treating Data {i}/{data_lenth} epoch {ep} " )
 
        data=dataset[i]
        inputs = data[0]
        label=data[1]
 
        
        out,hisory=forward_propagate(weights, inputs)
       
        delta= error (out,label)
       
        weights = back_propagate(weights,delta,hisory,lr=0.01)
        
        
    return weights


def dater_counter (A,X,trans,n):
    
    col=[]
    max_step= E
    dater_steps=[]
    counter_steps=[]
    
    
    
    for i in range (len(X)):
         counter_steps.append([])
         
     
    for i in range (n):
        
        An=Matpower(A,i)
        next_step=Matmul(An,X)
        dater_steps.append(next_step )     
        if max(next_step)>max_step: max_step=max(next_step)
    
    for i in range (1,max_step+1):      
        col.append("Step "+str(i))
    
    
    for step in dater_steps :
        for j in range (len(step)):
             counter_steps[j].append (step[j])
             
             
    zero_data = np.zeros(shape=(len(trans),len(col)))
    df = pd.DataFrame(zero_data, columns=col,index=index)

    for i in range (len (counter_steps)) :
        for j in counter_steps[i] :
            
            df.iloc[i][(j-1)]=1
            
  
    return dater_steps ,counter_steps,df 


def plan (dataframe,step):
    
    enabled_trans=[]
    
    for i in dataframe.index: 
        if (dataframe["Step "+str(step)][i])==1:
            enabled_trans.append(i)
            
    print(f"enabled transitions in step {step} : {enabled_trans}")
    return(enabled_trans)
       


#%%training MPMLP
train=1


if train==1:

    epochs=10
    weights=initialize_layer(4 ,4)
    oldweight=copy.deepcopy(weights)
    dataset=dataset_creator_states(10)

    for ep in range (epochs):
        weights=training(dataset,weights,ep)
        
    cleaned_weights=(clean(weights))
  

#%% tesing MPMLP

test=1

if test ==1 :
    X= [random.randint(-1, 4) for i in range(4)]
    
    print(X)
    
    
    
    A_0= Matmul(hAs,hA1)
    
    AX= Matmul(A_0,X)   
    AY= Matmul(weights,X)
    AZ= Matmul(cleaned_weights,X)
    
        
    print("\n old weights")    
    for i in oldweight:
        print(i)
        
    print("\n Raw weights ")    
    for i in weights:
        print(i)
        
    print("\n Cleaned weights ")    
    for i in cleaned_weights:
        print(i)
        
        
    print(" \n Reference Matrix " ) 
    for i in A_0:
        print(i)
        
       
    print(" \n Outputs comparaison " ) 
       
    print(AY) 
    print(AZ)
    print(AX)
    
    

    
#%% ordinary MLP Matrix multiplication8 
index=["X1","X2","X3","X4"]
dater ,counter,summary =dater_counter (cleaned_weights,X,index,10)

print(" \n Dater  : ")
for i in dater :
    print(i)
    
print(" \n Counter  : ")  

for i in counter :
    print(i)
    
print(" \n Summary : ")      
print(summary)

plan (summary,4)

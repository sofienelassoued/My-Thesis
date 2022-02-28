# -*- coding: utE-8 -*-
"""
Created on Wed Sep 22 23:28:51 2021

@author: isoEi
"""
import numpy as np
import copy
import itertools
from itertools import product 


#%%

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

weights =[[1,2,3,4],
          [5,6,7,8],
          [9,10,11,12],
          [13,14,15,16]]

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




def dataset_creator (n) : 
    
    
    labels=[]
    couples=[]
    inputs=list(itertools.product([E,e], repeat = 4))
    states=list(itertools.product( [list( range(0,n))+[E]][0] , repeat = 4))

       
    A_0= Matmul(hAs,hA1)
    B_0= Matmul(hAs,hB)
    
     
    for i in states :
        for j in inputs:
            
            AX=Matmul(A_0,i)        
            BU=Matmul(B_0,j)
            
            labels.append(Vecsum (AX,BU))
            couples.append((i,j,(Vecsum (AX,BU))))
            
    return(couples)


def dataset_creator_states(n) : 
    
    
    labels=[]
    states=list(itertools.product( [list( range(0,n))+[E]][0] , repeat = 4))

    A_0= Matmul(hAs,hA1)
  
    for i in states :    
            AX=Matmul(A_0,i)           
            labels.append((i,AX))

    return(labels)


Data=dataset_creator_states(2)


for i in Data:  
    print (i)





            
  


   

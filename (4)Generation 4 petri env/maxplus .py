# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:28:51 2021

@author: isofi
"""
import numpy as np
import copy

f=-1*np.inf
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






A = [[f,f,f,f,f,f,f,f],
    [f,a,f,f,f,f,f,f],
    [f,f,b,f,f,f,f,f],
    [f,f,f,c,f,f,f,f],
    [f,f,f,f,f,f,f,f],
    [f,f,f,f,x,f,f,f],
    [f,f,f,f,f,z,f,f],
    [f,f,f,f,f,f,w,f]]


B = [[f,f,f,f,f,f,f,f],
    [f,a,f,f,f,f,f,f],
    [f,f,b,f,f,f,f,f],
    [f,f,f,c,f,f,f,f],
    [f,f,f,f,f,f,f,f],
    [f,f,f,f,x,f,f,f],
    [f,f,f,f,f,z,f,f],
    [f,f,f,f,f,f,w,f]]


C= [[f,f,f,f,f,f],
    [5,f,f,f,f,f],
    [3,f,f,f,f,f],
    [f,2,f,f,5,f],
    [f,1,4,f,f,f],
    [f,f,8,2,4,f]]




def Max(a,b):
    
    if a == f : return b
    elif b == f: return a
    
    try :return max  (a,b )
    except :return ("Max (" + str(a) +","+ str(b) +")" )
     
    

def Plus(a,b):
    if a == f : return a
    elif b == f : return b
    
    try :return int  (a+b )
    except :return (str(a) +"+"+ str(b)  )
    


def Matmul(A,B)  :
    
    result=[]
    for i in range (len (A) ):
        result.append([f]* len(A[0]))
 
    # iterate through rows of X
    for i in range(len(A)):
        
        # iterate through columns of Y
        for j in range(len(B[0])):
           # iterate through rows of Y
            for k in range(len(B)):
                
                result[i][j]=Max( result[i][j] , Plus (A[i][k] , B[k][j]))
  
    return(result)

  

def Matpower(A,n): 
    result=A
    for i in range (1,n):
        result=Matmul(A,result)
        
    return(result)


res=Matpower(A,1000)


for r in res:
        print(r)
     
   
   
   
   

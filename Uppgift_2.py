# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:56:17 2018

@author: Bjorn
"""
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style as stl


## Ger en matris x som varje rad är värderna för en punkt 3 *30 matris och en vektor y som ger label för varje rad

def importerino(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    matrix = []
    
    for i in range(0, len(content)):
        matrix.append(content[i].split())
        
    rows = len(matrix)
    cols = len(matrix[0])  
    
    for i in range(0,rows):
        for j in range(1,cols):
           matrix[i][j] = matrix[i][j].split(':', 1)[-1]
    
    y = np.zeros(rows)
    x = np.ones((rows,cols))
    
    for i in range(0,rows):
        label = float(matrix[i][0])
        if label == 1:
            y[i] = label
            
        for j in range(1,cols):
            x[i,j] = float(matrix[i][j])
            
            
    return x,y



    
def get_guess_step(w,x):
    if np.dot(w,x)>0:
        return 1
    else: 
        return 0
    
def get_guess_sigmoid(w,x):
    return 1/(1+np.exp(-np.dot(w,x)))
    
def normalize(x):
    x_norm = np.zeros((len(x),len(x[0])))
    for i in range(1,len(x[0])):
        x_norm[:,i] = (x[:,i] - np.min(x[:,i]))/(np.max(x[:,i])-np.min(x[:,i]))   
    x_norm[:,0] = 1
    return x_norm







def perceptron_algorithm(x,y, learning_rate,isStep):
    w = np.zeros(len(x[0]))
    errors=1337
    old_error=1338
    new_error =1337
    while errors>0:
        errors=0
        for i in range(len(x)): 
            guess=0
            if(isStep):
                guess= get_guess_step(w,x[i])
            else :
                guess=get_guess_sigmoid(w,x[i])
            #guess = get_guess_sigmoid(w,x[i])    
            error= y[i] - guess
            if(abs(error)>0.5):
                errors+=1
                
            delta_w = error*x[i]*learning_rate
            w += delta_w
        new_error=errors
        if new_error<old_error:
            print("total errors",errors)
            old_error=errors
            
        
        
    return w

    
def plot(x1,y1,isStep,learning_rate):    
    w = perceptron_algorithm(x1,y1,learning_rate,isStep)
    x = np.arange(0, 1, 0.001);
    y = x*w[1]/w[2]+w[0]/w[2]
    plt.plot(x, -y)
    if isStep:
        print("\nStep:")
    else:
        print ("\nSigmoid:")
        
    print("Weights: W0: ",w[0]," W1: ", w[1], " W2: ",w[2],"\n")
    print("y=kx+m där m= ",-w[0]/w[2]," och k= ",-w[1]/w[2],"\n")
  




learning_rate=float(input("Which learning rate rate you you like to have?  (lower than 1 will take some time)"))

x,y = importerino('salammbo.txt')
#print(x)

xn = normalize(x)
#perceptron(xn,y)

plt.figure(2)
plt.scatter(xn[:,1],xn[:,2])
plot(xn,y,True,learning_rate)
plt.title("Step")

plt.figure(1)
plt.scatter(xn[:,1],xn[:,2])
plot(xn,y,False,learning_rate)
plt.title("Sigmoid")





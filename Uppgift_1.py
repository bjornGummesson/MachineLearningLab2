#coding:utf8
#!/usr/bin/env python
from numpy import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style as stl




"""

"""




stl.use("fivethirtyeight")


def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        x = [line.split()[0] for line in lines]
        y = [line.split()[1] for line in lines]
    return x,y 

def string_to_float(x,y):
    x_float = []
    y_float = []
    
    for i in range (len(x)):
        x_float.append(float(x[i]))
        y_float.append(float(y[i]))
    return x_float, y_float
    
def computeErrorForLineGivenPoints(b, m, x, y):
    """
    ser bra ut
    """
    totalError = 0
    for i in range(0, len(x)):
        totalError += (y[i] - (m * x[i] + b)) ** 2
    return totalError / float(len(x))

def get_gradiant(b_current, m_current, x,y):
        b_gradient = -1* (y - ((m_current*x) + b_current))
        m_gradient = -1* x* (y - ((m_current * x) + b_current))
        return b_gradient, m_gradient

def stepGradient_batch(b_current, m_current, x,y, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(x))
    for i in range(0, len(x)):
        bg,mg = get_gradiant(b_current,m_current,x[i],y[i])
        b_gradient += (2/N) * bg
        m_gradient += (2/N) * mg
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return new_b, new_m




def normalize(x):
    x_norm=[]
    for i in range (len(x)):
        x_norm.append((x[i]-min(x))/(max(x)-min(x)))

    x_norm_np = np.array(x_norm, dtype=np.float64)
    return x_norm_np    

def gradient_descent_stochastics(b_current, m_current, x,y,learningRate):
    bg,mg = get_gradiant(b_current,m_current,x,y)
    new_b = b_current - (learningRate * bg)
    new_m = m_current - (learningRate * mg)
    return new_b, new_m

    
def printus_maximus2dot0_batch(x,y,isEnglish): 

    m,b=0,0
    x_norm=normalize(x)
    y_norm=normalize(y)
    error = computeErrorForLineGivenPoints(b,m,x_norm,y_norm)
    last_error=error+100
    while error<last_error: 
        b,m=stepGradient_batch(b,m,x_norm,y_norm,0.01) 
        last_error=error
        error = computeErrorForLineGivenPoints(b,m,x_norm,y_norm)
    print("for y=mx+b")
       # print(error)
    if(isEnglish):
        print("English:\n")
    else:
        print("French:\n")    
    print("Batch: M: ",m, " B: ",b,"\n \n") 
    
    
    regression_line=[(m*x_norm)+b for x_norm in x_norm]
    plt.scatter(y_norm,x_norm)
    plt.plot(x_norm,regression_line)
    
def printus_maximus2dot0_stoch(x,y,isEnglish): 
    
    m,b=0,0
    x_norm=normalize(x)
    y_norm=normalize(y)
    error = computeErrorForLineGivenPoints(b,m,x_norm,y_norm)
    last_error=error+100
    while error<last_error: 
        for i in range(len(x)):
            b,m = gradient_descent_stochastics(b,m,x_norm[i],y_norm[i],0.001)
        last_error=error
        error = computeErrorForLineGivenPoints(b,m,x_norm,y_norm)
    print("for y=mx+b")
        #print(error)
    if(isEnglish):
        print("English:\n")
    else:
        print("French:\n")  
   
    print("Stoch: M: ",m, " B: ",b,"\n \n") 
    
    regression_line=[(m*x_norm)+b for x_norm in x_norm]
    plt.scatter(y_norm,x_norm)
    plt.plot(x_norm,regression_line)
    
    

xf,yf = read_file("french.plot")
xe,ye = read_file("english.plot") 

xf,yf= string_to_float(xf,yf)
xe,ye= string_to_float(xe,ye)
    
printus_maximus2dot0_stoch(xf,yf,False)
printus_maximus2dot0_batch(xf,yf,False)

printus_maximus2dot0_stoch(xe,ye,True)
printus_maximus2dot0_batch(xe,ye,True)



#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

def degrau(n,n0=0):
    n = n-n0
    u = np.array([])
    for i in n:
        if(i >= 0):
            u = np.append(u,1.0)
        else:
            u = np.append(u,0.0)
    return u

    

t = np.arange(0,0.2,0.001)

#Letra A
Vi = 1
R = (1 + 0.5*np.cos(20*pi*t))*degrau(t)

saidaA = np.array(-R*Vi)
saidaB = np.array(-R*Vi)*degrau(t,0.05)

#Letra A
fig1, ax1 = plt.subplots()
ax1.plot(t,saidaA,"c-",linewidth = 1,label = "Exercício 1 - Letra A - Vo(t)")
ax1.legend()
fig1.show()
plt.show()

#Letra B
fig1, ax1 = plt.subplots()
ax1.plot(t,saidaB,"c-",linewidth = 1,label = "Exercício 1 - Letra B - Vo(t)")
ax1.legend()
fig1.show()
plt.show()


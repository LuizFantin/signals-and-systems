
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



n = np.arange(31)

#Letra A
entradaA = degrau(n) - degrau(n,8)
sistemaA = np.sin((2*pi/8)*n)*(degrau(n) - degrau(n,8))
saidaA = np.convolve(entradaA,sistemaA)

saidaA = np.array(saidaA)
saidaA = np.array(saidaA[:31])

#Letra B
entradaB = np.sin((2*pi/8)*n)*(degrau(n) - degrau(n,8))
sistemaB = -np.sin((2*pi/8)*n)*(degrau(n) - degrau(n,8))
saidaB = np.convolve(entradaB,sistemaB)

saidaB = np.array(saidaB)
saidaB = np.array(saidaB[:31])

#Letra A
fig1, ax1 = plt.subplots()
ax1.stem(n,entradaA,linefmt="r-",label = "Exercício 2 - Letra A - x[n]")
ax1.legend()
fig1.show()
plt.show()
fig1, ax1 = plt.subplots()
ax1.stem(n,sistemaA,linefmt="r-",label = "Exercício 2 - Letra A - h[n]")
ax1.legend()
fig1.show()
plt.show()
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaA,linefmt="r-",label = "Exercício 2 - Letra A - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra B
fig1, ax1 = plt.subplots()
ax1.stem(n,entradaB,linefmt="r-",label = "Exercício 2 - Letra B - x[n]")
ax1.legend()
fig1.show()
plt.show()
fig1, ax1 = plt.subplots()
ax1.stem(n,sistemaB,linefmt="r-",label = "Exercício 2 - Letra B - h[n]")
ax1.legend()
fig1.show()
plt.show()
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaB,linefmt="r-",label = "Exercício 2 - Letra B - y[n]")
ax1.legend()
fig1.show()
plt.show()

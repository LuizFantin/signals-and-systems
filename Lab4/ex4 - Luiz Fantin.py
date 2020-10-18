
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

def degrau(n,n0=0):
    nd = n-n0
    u = np.array([])
    for i in nd:
        if(i >= 0):
            u = np.append(u,1.0)
        else:
            u = np.append(u,0.0)
    return u

def impulso(n,n0=0):
    n = n-n0
    u = np.array([])
    for i in n:
        if(i == 0):
            u = np.append(u,1.0)
        else:
            u = np.append(u,0.0)
    return u

n = np.arange(-2,21)
entrada = degrau(n)

#Letra A
sistemaA = ((1/2)**2)*degrau(n)
saidaA = np.convolve(entrada,sistemaA)
saidaA = np.array(saidaA[2:25])
#Letra B
sistemaB = impulso(n) - impulso(n,1)
saidaB = np.convolve(entrada,sistemaB)
saidaB = np.array(saidaB[2:25])
#Letra C
sistemaC = ((-1.0)**n)*(degrau(n,-2) - degrau(n,3))
saidaC = np.convolve(entrada,sistemaC)
saidaC = np.array(saidaC[2:25])
#Letra D
sistemaD = degrau(n)
saidaD = np.convolve(entrada,sistemaD)
saidaD = np.array(saidaD[2:25])
#Letra E
sistemaE = (-n)*degrau(n)
saidaE = np.convolve(entrada,sistemaE)
saidaE = np.array(saidaE[2:25])
#Letra F
sistemaF = np.sin((1/12)*pi*n)*degrau(n,3)
saidaF = np.convolve(entrada,sistemaF)
saidaF = np.array(saidaF[2:25])

#Letra A
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaA,linefmt="r-",label = "Exercício 4 - Letra A - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra B
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaB,linefmt="r-",label = "Exercício 4 - Letra B - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra C
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaC,linefmt="r-",label = "Exercício 4 - Letra C - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra D
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaD,linefmt="r-",label = "Exercício 4 - Letra D - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra E
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaE,linefmt="r-",label = "Exercício 4 - Letra E - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Letra F
fig1, ax1 = plt.subplots()
ax1.stem(n,saidaF,linefmt="r-",label = "Exercício 4 - Letra F - y[n]")
ax1.legend()
fig1.show()
plt.show()


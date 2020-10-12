
#Luiz Fantin Neto - 2018102590

import numpy as np
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



n = np.arange(-10, 101)
sistema = ((3/4)**n)*degrau(n)
entrada = degrau(n)
saida = np.convolve(entrada,sistema)

saida = np.array(saida[10:121])


fig1, ax1 = plt.subplots()
ax1.stem(n,saida,linefmt="r-",label = "Exercício 2 - y[n]")
ax1.legend()
fig1.show()
plt.show()

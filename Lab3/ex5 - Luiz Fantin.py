
#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def entrada(n):
    if n>=0 and n<=4:
        return 1
    else: 
        return 0

def sistema(n, alpha):
    if n>=0 and n<=6:
        return alpha**n
    else: 
        return 0

n = np.arange(0, 21)
alpha = 0.5
entrada = np.array([entrada(elem) for elem in n])
sistema = np.array([sistema(elem,alpha) for elem in n])

saida = np.convolve(entrada,sistema)
saida = np.array(saida[:21])

fig1, ax1 = plt.subplots()
ax1.stem(n,saida,linefmt="r-",label = "Exercício 5 - y[n]")
ax1.legend()
fig1.show()
plt.show()

#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def entrada(n):
    if n == 1:
        return 2
    elif n == 0:
        return 0.5
    else:
        return 0

def sistema(n):
    if n>=0 and n<=2:
        return 1
    else:
        return 0


n = np.arange(4)

entrada = np.array([entrada(elem) for elem in n])

sistema = np.array([sistema(elem) for elem in n])

def conv(entrada, sistema, n):
    saida = np.array([0 for elem in n])
    for i in range(0, 3):
        saida = np.sum([np.roll(sistema, i)*entrada[i], saida], axis=0)
    return saida


saida = conv(entrada, sistema, n)

fig1, ax1 = plt.subplots()
ax1.stem(n,saida,linefmt="r-",label = "Exercício 1 - Letra A - y[n]")
ax1.legend()
fig1.show()
plt.show()

saida2 = np.convolve(entrada, sistema)
saida2 = np.array(saida2[:4])

fig1, ax1 = plt.subplots()
ax1.stem(n,saida2,linefmt="r-",label = "Exercício 1 - Letra B - y[n]")
ax1.legend()
fig1.show()
plt.show()
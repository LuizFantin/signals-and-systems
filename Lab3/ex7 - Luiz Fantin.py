
#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

amostragem = 0.0001


def entrada(t):
    if t>0 and t<1:
        return 1
    else:
        return 0

def sistema(t):
    if t>0 and t<2:
        return t
    else:
        0

t = np.arange(0, 10, amostragem)
entrada = np.array([entrada(elem) for elem in t])
sistema = np.array([sistema(elem) for elem in t])

saida = np.convolve(entrada,sistema)*amostragem
saida = saida[:t.size]

fig1, ax1 = plt.subplots()
ax1.plot(t,saida,linewidth = 1,label = "Exercício 7 - y[n]")
ax1.legend()
fig1.show()
plt.show()

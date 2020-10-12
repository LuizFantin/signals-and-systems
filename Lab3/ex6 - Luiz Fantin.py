
#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

amostragem = 0.0001
RC = 1

def degrau(n,n0=0):
    n = n-n0
    u = np.array([])
    for i in n:
        if(i >= 0):
            u = np.append(u,1.0)
        else:
            u = np.append(u,0.0)
    return u



t = np.arange(0, 5*RC, amostragem)
entrada = degrau(t)-degrau(t,1)
sistema = degrau(t)*np.exp(-t)

#y(0) é em 0
#y(5*RC) é em 5*RC/Ts (t.size)
saida = np.convolve(entrada, sistema)*amostragem
saida = saida[:t.size]

fig1, ax1 = plt.subplots()
ax1.plot(t,saida,linewidth = 1,label = "Exercício 6 - y[n]")
ax1.legend()
fig1.show()
plt.show()

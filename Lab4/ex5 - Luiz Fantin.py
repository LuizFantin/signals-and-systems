
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
from scipy import signal
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


def trem(t,T):
    trem = np.arange(-10, 10, 1)
    for pos in np.arange(t.size):
        if(t[pos] % T != 0):
            trem[pos] = 0
        else:
            trem[pos] = 1
    return trem

T1 = 4
T2 = 2
T3 = 3/2
T4 = 1
t = np.arange(-10, 10, 1)
sistema = np.array([elem+1 if -1 <= elem <= 0 else -elem+1 if 0 < elem <= 1 else 0 for elem in t])

#Letra A
entrada1 = trem(t, T1)
saida1 = np.convolve(entrada1,sistema)
saida1 = np.array(saida1[10:30])

#Letra B
entrada2 = trem(t, T2)
saida2 = np.convolve(entrada2,sistema)
saida2 = np.array(saida2[10:30])

#Letra C
entrada3 = trem(t, T3)
saida3 = np.convolve(entrada3,sistema)
saida3 = np.array(saida3[10:30])

#Letra D
entrada4 = trem(t, T4)
saida4 = np.convolve(entrada4,sistema)
saida4 = np.array(saida4[10:30])


#Plotando Letra A
fig1, ax1 = plt.subplots(3,1)
ax1[0].stem(t, entrada1, 'c-', label="")
ax1[0].set_xlabel("t")
ax1[0].set_ylabel("x(t)")

ax1[1].plot(t, sistema, 'c-', label="")
ax1[1].set_xlabel("t")
ax1[1].set_ylabel("h(t)")

ax1[2].plot(t, saida1, 'c-', label="")
ax1[2].set_xlabel("t")
ax1[2].set_ylabel("y(t)")

plt.tight_layout()
plt.show()

#Plotando Letra B
fig1, ax1 = plt.subplots(3,1)
ax1[0].stem(t, entrada2, 'c-', label="")
ax1[0].set_xlabel("t")
ax1[0].set_ylabel("x(t)")

ax1[1].plot(t, sistema, 'c-', label="")
ax1[1].set_xlabel("t")
ax1[1].set_ylabel("h(t)")

ax1[2].plot(t, saida2, 'c-', label="")
ax1[2].set_xlabel("t")
ax1[2].set_ylabel("y(t)")

plt.tight_layout()
plt.show()

#Plotando Letra C
fig1, ax1 = plt.subplots(3,1)
ax1[0].stem(t, entrada3, 'c-', label="")
ax1[0].set_xlabel("t")
ax1[0].set_ylabel("x(t)")

ax1[1].plot(t, sistema, 'c-', label="")
ax1[1].set_xlabel("t")
ax1[1].set_ylabel("h(t)")

ax1[2].plot(t, saida3, 'c-', label="")
ax1[2].set_xlabel("t")
ax1[2].set_ylabel("y(t)")

plt.tight_layout()
plt.show()

#Plotando Letra D
fig1, ax1 = plt.subplots(3,1)
ax1[0].stem(t, entrada4, 'c-', label="")
ax1[0].set_xlabel("t")
ax1[0].set_ylabel("x(t)")

ax1[1].plot(t, sistema, 'c-', label="")
ax1[1].set_xlabel("t")
ax1[1].set_ylabel("h(t)")

ax1[2].plot(t, saida4, 'c-', label="")
ax1[2].set_xlabel("t")
ax1[2].set_ylabel("y(t)")

plt.tight_layout()
plt.show()

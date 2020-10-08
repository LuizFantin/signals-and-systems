
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
from numpy import e
import scipy as sp
import matplotlib.pyplot as plt


t = np.linspace(0,3,700)
x1 = 10*e**(-2*t)*np.sin(30*pi*t + (pi/2))
x2 = 10*e**(-2*t)*np.sin(30*pi*t)
x3 = 10*e**(2*t)*np.sin(30*pi*t + (pi/2))


fig1, ax1 = plt.subplots()
ax1.plot(t,x1,'c-', linewidth = 1 ,label = "Função A")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
ax1.legend()
fig1.show()

plt.show()
fig1, ax1 = plt.subplots()
ax1.plot(t,x2,'c-', linewidth = 1 ,label = "Função B")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
ax1.legend()
fig1.show()

plt.show()
fig1, ax1 = plt.subplots()
ax1.plot(t,x3,'c-', linewidth = 1 ,label = "Função C")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
ax1.legend()
fig1.show()

plt.show()

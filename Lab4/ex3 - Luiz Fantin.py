
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

t = np.arange(0,4,0.001)


Vi = np.cos(pi*t)
   
Vout = np.array([0.5 if elem > 0.5 else -0.5 if elem < -0.5 else elem for elem in Vi])

Vi2 = 0.3*np.cos(pi*t)
   
Vout2 = np.array([0.5 if elem > 0.5 else -0.5 if elem < -0.5 else elem for elem in Vi2])


fig1, ax1 = plt.subplots()
ax1.plot(t,Vi,"c-",linewidth = 1,label = "Exercício 3 - I - Vi(t)")
ax1.plot(t,Vout,"c-",linewidth = 1,label = "Exercício 3 - I - Vout(t)")
ax1.legend()
fig1.show()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t,Vi2,"c-",linewidth = 1,label = "Exercício 3 - II - Vi(t)")
ax2.plot(t,Vout2,"c-",linewidth = 1,label = "Exercício 3 - II - Vout(t)")
ax2.legend()
fig2.show()
plt.show()

print("\nLetra A: Não é linear\n")
print("\nLetra B: É invariante no tempo\n")



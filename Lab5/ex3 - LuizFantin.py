
#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift
from scipy import signal


frequencia = 2/3                            
Ts = 1/frequencia                         
w0 = 2*pi*frequencia                       
fim = Ts/100   	                    

t = np.arange(-3,3,fim)             
x = signal.sawtooth(t*w0)*0.75 + 0.25     

X = fft(x)/len(x)                   
w = fftfreq(len(t), d=(1/Ts)*fim)   

Xd = fftshift(X)                    
w = fftshift(w)

ModX = np.abs(Xd)                   
phasX = np.angle(Xd)                
phasX[ModX < 0.0001] = 0

xr = ifft(X)*len(x)                 
xr = np.real(xr)                    


#Plotar graficos

fig, ax1 = plt.subplots(2, 1)
ax1[0].plot(t, x, 'c-', linewidth=2, label="x(t)")
ax1[0].set_ylabel("Amplitude")
ax1[0].set_xlabel("tempo [s]")
ax1[0].grid(True)
ax1[0].set_title('x[n] - Original - Exercicio 3')

ax1[1].plot(t, xr, 'c-', linewidth=2, label="xr(t)")
ax1[1].set_ylabel("Amplitude")
ax1[1].set_xlabel("tempo [s]")
ax1[1].grid(True)
ax1[1].set_title('x(t) - Recuperado - Exercicio 3')

fig.tight_layout()

fig1, ax = plt.subplots(2, 1)
ax[0].stem(w, ModX, linefmt='c-', use_line_collection=True)
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("k")
ax[0].grid(True)
ax[0].set_title('|X[k]| - Exercicio 3')

ax[1].stem(w, phasX, linefmt='c-', use_line_collection=True)
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("k")
ax[1].grid(True)
ax[1].set_title('angle(X[k]) - Exercicio 3')

fig1.tight_layout()
plt.show()


#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift


def rampa(t,d,tam):
    x = np.array([])
    ponto = 0
    for i in range(len(t)):
        if (t[i]<d):
            x = np.append(x,0)
        else:
            x = np.append(x,ponto)
            ponto = ponto + tam
    return x

# Criando o vetor de tempo
# Frequencia de amostragem
wam = 10*100 # 10 é a freq. máx do sinal
Tam = (2*pi)/wam
t = np.arange(-10,10,Tam)

p = np.cos(10*t)
x = 0.2*(rampa(t,5,Tam)-2*rampa(t,0,Tam)+rampa(t,-5,Tam))

y = p*x
# N - tamanho da DTFS
N = 2**12

# calculando a FT
X = (Tam*N)*fft(x,N)/N
Y = (Tam*N)*fft(y,N)/N


#criando o vetor de frequencia
w = fftfreq(len(X), d=(Tam))*(2*pi)

#saida

# Os indices de frequencia são mudados de 0 a N-1 para -N/2 + 1 a N/2
# posicionando a freq. zero no meio do gráfico
wd = fftshift(w)
Xd = fftshift(X)
Yd = fftshift(Y)

# calculando o modulo - magnitude do espectro
ModX = np.abs(Xd)
ModY = np.abs(Yd)

fig, ax = plt.subplots(2,1)
ax[0].plot(wd, ModX, 'r-', linewidth=1, label="x(t)")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("frequencia [rad/s]")
ax[0].grid(True)
# ax1.legend()
ax[0].set_xlim(-40, 40)
ax[0].set_title('|X(e^jw)|')

ax[1].plot(t, x, 'c-', linewidth=2, label="x(t)")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("tempo [s]")
ax[1].grid(True)
ax[1].set_title('x(t)')

fig, ax = plt.subplots(2,1)
ax[0].plot(wd, ModY, 'r-', linewidth=1, label="y(t)")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("frequencia [rad/s]")
ax[0].grid(True)
# ax1.legend()
ax[0].set_xlim(-40, 40)
ax[0].set_title('|Y(e^jw)|')

ax[1].plot(t, y, 'c-', linewidth=2, label="y(t)")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("tempo [s]")
ax[1].grid(True)
ax[1].set_title('y(t)')

plt.show()


# Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift

def plot(k,x,letra):
    fig1, ax = plt.subplots()
    ax.stem(k,x, 'c-', label="|X[K]|")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("k")
    ax.grid(True)
    ax.set_title('|X[k]| - Letra '+letra)

    fig1.tight_layout()
    fig1.tight_layout()

    plt.show()

#usado na letra C
def sinal3(mod,phas):
    sinal = np.array([])

    for i in range(len(mod)):
        x = mod[i] * np.cos(phas[i]*pi/2)
        y = mod[i] * np.sin(phas[i]*pi/2)

        sinal = np.append(sinal,complex(x,y))
    return sinal

# criando o vetor de amostra com tamanho nper  periodos do sinal
N1 = 17 # periodo do sinal A
N2 = 21 # periodo do sinal B
N3 = 7 # periodo do sinal B
N4 = 7 # periodo do sinal B

nper = 1 # quantidade de periodos em x[n]

k1 = np.arange(0,nper*N1)
k2 = np.arange(0,nper*N2)
k3 = np.arange(0,nper*N3)
k4 = np.arange(0,nper*N4)

#sinal A
x1 = np.cos((6*pi/17)*k1)
x2 = np.cos((10*pi/21)*k2) + 1j*np.cos((4*pi/21)*k2)
x3Mod = np.array([0.0,0.0,0.0,1.0,1.0,0.0,0.0])
x3Phas = np.array([0.0,0.0,0.0,-1.0,1.0,0.0,0.0])
x3 = sinal3(x3Mod,x3Phas)
x4 = np.array([-0.5,1.0,0.0,0.0,0.0,0.0,1.0])

#criando o vetor de frequencia
w1 = fftfreq(len(k1), d=1/N1)
w2 = fftfreq(len(k2), d=1/N2)
w3 = fftfreq(len(k3), d=1/N3)
w4 = fftfreq(len(k4), d=1/N4)


# retornando o sinal ao dominio do tempo
xr1 = ifft(x1)*len(k1)
xr2 = ifft(x2)*len(k2)
xr3 = ifft(x3)*len(k3)
xr4 = ifft(x4)*len(k4)

xr1 = np.real(xr1) # ignorando os erros de arrendondamento do fft e ifft
xr2 = np.real(xr2) # ignorando os erros de arrendondamento do fft e ifft
xr3 = np.real(xr3) # ignorando os erros de arrendondamento do fft e ifft
xr4 = np.real(xr4) # ignorando os erros de arrendondamento do fft e ifft

#plotando os gráficos
plot(w1,xr1,letra='A')
plot(w2,xr2,letra='B')
plot(w3,xr3,letra='C')
plot(w4,xr4,letra='D')

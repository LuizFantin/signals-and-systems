
# Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift

def plot(y,x1,x2,letra):
    fig1, ax = plt.subplots(2, 1)
    ax[0].stem(y,x1, 'c-', label="|X[K]|")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("k")
    ax[0].grid(True)
    ax[0].set_title('|X[k]| - Letra '+letra)

    ax[1].stem(y,x2, 'c-', label="angle(X[k])")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("k")
    ax[1].grid(True)
    ax[1].set_title('angle(X[k]) - Letra '+letra)

    fig1.tight_layout()
    fig1.tight_layout()

    plt.show()

# criando o vetor de amostra com tamanho nper  periodos do sinal
N1 = 13 # periodo do sinal A
N2 = 21 # periodo do sinal B
N3 = 8 # periodo do sinal C
N4 = 5 # periodo do sinal D
N5 = 9 # periodo do sinal E

nper = 2 # quantidade de periodos em x[n]
period = aux = 3  #usado na letra C D e E

n1 = np.arange(0,nper*N1)
n2 = np.arange(0,nper*N2)
n3 = np.arange(-13, (period * N3) - 13)  
n4 = np.arange(-10, (period * N4) - N4)
n5 = np.arange(-10, (period * N5) - N5 - 1)

#sinal A
x1 = np.cos(((6*pi/13)*n1)+(pi/6))

#sinal B
x2 = np.sin((4*pi/21)*n2) + np.cos((10*pi/21)*n2) + 1   

#sinal C
x3 = np.zeros(N3)            
x3[3] = -1; x3[7] = 1

i = 0; aux = x3
while (i < period - 1):
    aux = list(aux) + list(x3)
    i += 1
x3 = aux

#sinal D
x4 = np.zeros(N4)                                
i = 1
while (i < N4):
    x4[i] = (1 / (N4 - i))
    i += 1
i = 0; aux = x4
while (i < period):
    aux = list(aux) + list(x4)
    i += 1
x4=aux

#sinal E
x5 = np.zeros(N5)                                 
i = 0
while (i < N5):
    if (i < 4):
        x5[i] = -1
    if (i > 4):
        x5[i] = 1
    i += 1
i = 1;aux = x5
while (i < period):
    aux = list(aux) + list(x5)
    i += 1
x5=aux

# calculando a DTFS
X1 = fft(x1)/len(x1)
X2 = fft(x2)/len(x2)
X3 = fft(x3)/len(x3)
X4 = fft(x4)/len(x4)
X5 = fft(x5)/len(x5)

#criando o vetor de frequencia
w1 = fftfreq(len(n1), d=1/N1)
w2 = fftfreq(len(n2), d=1/N2)
w3 = fftfreq(len(n3), d=1/N3)
w4 = fftfreq(len(n4), d=1/N4)
w5 = fftfreq(len(n5), d=1/N5)

# Os indices de frequencia sÃ£o mudados de 0 a N-1 para -N/2 + 1 a N/2
# posicionando a freq. zero no meio do grÃ¡fico
Xd1 = fftshift(X1)
Xd2 = fftshift(X2)
Xd3 = fftshift(X3)
Xd4 = fftshift(X4)
Xd5 = fftshift(X5)

wd1 = fftshift(w1)
wd2 = fftshift(w2)
wd3 = fftshift(w3)
wd4 = fftshift(w4)
wd5 = fftshift(w5)

# calculando o modulo - magnitude do espectro
ModX1 = np.abs(Xd1)
ModX2 = np.abs(Xd2)
ModX3 = np.abs(Xd3)
ModX4 = np.abs(Xd4)
ModX5 = np.abs(Xd5)

# calculando a fase do espectro
phasX1 = np.angle(Xd1)
phasX2 = np.angle(Xd2)
phasX3 = np.angle(Xd3)
phasX4 = np.angle(Xd4)
phasX5 = np.angle(Xd5)

# devido a erros de arredondamentos numericos da fft devemos filtrar os sinais muito pequenos!
phasX1[ModX1 < 0.00001] = 0 
phasX2[ModX2 < 0.00001] = 0 
phasX3[ModX3 < 0.00001] = 0 
phasX4[ModX4 < 0.00001] = 0 
phasX5[ModX5 < 0.00001] = 0 


#plotando os gráficos
plot(wd1,ModX1,phasX1,letra='A')
plot(wd2,ModX2,phasX2,letra='B')
plot(wd3,ModX3,phasX3,letra='C')
plot(wd4,ModX4,phasX4,letra='D')
plot(wd5,ModX5,phasX5,letra='E')

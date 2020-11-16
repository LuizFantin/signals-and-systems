
#Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift

def trem(t,T):
    train = np.array([])
    for i in range(len(t)):
        if(t[i] % T == 0):
            train = np.append(train,1)
        else:
            train = np.append(train,0)
    return train

def degrau(n,n0=0):
    n = n-n0
    u = np.array([])
    for i in n:
        if(i >= 0):
            u = np.append(u,1.0)
        else:
            u = np.append(u,0.0)
    return u

#Letras A e B
# Criando o vetor de tempo
# Frequencia de amostragem
wam = 20*100 # 10 é a freq. máx do sinal
Tam = (2*pi)/wam
t = np.arange(-10,10,Tam)

#Letra C
wamc = 121*350  # - 
TamC = (2 * pi) / wamc    
tc = np.arange(-1, 1, TamC)

#Letra D
wamd = 2  
TamD = (2 * pi) / wamd  
td = np.arange(-10, 10 + Tam/pi, TamD/pi)

p = np.cos(10*t)
xa1 = degrau(t) - degrau(t,2)
xa2 = degrau(t) - degrau(t,1)
xa3 = degrau(t) - degrau(t,0.35)

xb = np.exp(-t)*degrau(t)
xc = np.sin(350*tc)
xd = trem(td,10)

# N - tamanho da DTFS
N = 2**12

# calculando a FT
X1 = (Tam*N)*fft(xa1,N)/N
X2 = (Tam*N)*fft(xa2,N)/N
X3 = (Tam*N)*fft(xa3,N)/N

Xb = (Tam*N)*fft(xb,N)/N
Xc = (TamC * N) * fft(xc, N)/N
Xd = (TamD * N) * fft(xd, N)/N


#criando o vetor de frequencia
w = fftfreq(len(X1), d=(Tam))*(2*pi)
wb = fftfreq(len(Xb), d=(Tam))*(2*pi)
wc = fftfreq(len(Xc), d=(TamC)) * (2 * pi)
wD = fftfreq(len(Xd), d=(TamD)) * (2 * pi)

#saida

# Os indices de frequencia são mudados de 0 a N-1 para -N/2 + 1 a N/2
# posicionando a freq. zero no meio do gráfico
wd = fftshift(w)
wdb = fftshift(wb)
wdc = fftshift(wc)
wdD = fftshift(wD)

Xd1 = fftshift(X1)
Xd2 = fftshift(X2)
Xd3 = fftshift(X3)
Xdb = fftshift(Xb)
Xdc = fftshift(Xc)
XdD = fftshift(Xd)

# calculando o modulo - magnitude do espectro
ModX1 = np.abs(Xd1)
ModX2 = np.abs(Xd2)
ModX3 = np.abs(Xd3)
ModXb = np.abs(Xdb)
ModXc = np.abs(Xdc)
ModXd = np.abs(XdD)

# calculando o modulo - magnitude do espectro
PhasX1 = np.angle(Xd1)
PhasX2 = np.angle(Xd2)
PhasX3 = np.angle(Xd3)
PhasXb = np.angle(Xdb)
PhasXc = np.angle(Xdc)
PhasXd = np.angle(XdD)

#Letra A
fig, ax = plt.subplots(3,1)
ax[0].plot(wd, ModX1,wd, ModX2,wd, ModX3,'r-', linewidth=1)
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("frequencia [rad/s]")
ax[0].grid(True)
# ax1.legend()
ax[0].set_xlim(-50, 50)
ax[0].set_title('Letra A - |X(e^jw)|')

ax[1].plot(wd, PhasX1 ,wd, PhasX2 ,wd, PhasX3 ,'r-', linewidth=1)
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("frequencia [rad/s]")
ax[1].grid(True)
# ax1.legend()
ax[1].set_xlim(-10, 10)
ax[1].set_title('Letra A - angle(X(e^jw)')

ax[2].plot(t, xa1,t,xa2,t,xa3, 'c-', linewidth=2, label="x(t)")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlabel("tempo [s]")
ax[2].grid(True)
ax[2].set_title('Letra A - x(t)')

#Letra B
fig, ax = plt.subplots(3,1)
ax[0].plot(wdb, ModXb, 'r-', linewidth=1)
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("frequencia [rad/s]")
ax[0].grid(True)
# ax1.legend()
ax[0].set_xlim(-20, 20)
ax[0].set_title('Letra B - |X(e^jw)|')

ax[1].stem(wdb, PhasXb)
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("frequencia [rad/s]")
ax[1].grid(True)
# ax1.legend()
ax[1].set_xlim(-20, 20)
ax[1].set_title('Letra B - angle(X(e^jw)')

ax[2].plot(t, xb, 'c-', linewidth=2, label="x(t)")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlabel("tempo [s]")
ax[2].grid(True)
ax[2].set_title('Letra B - x(t)')


#Letra C
fig, ax = plt.subplots(3,1)

ax[0].plot(wdc, ModXc, 'r-',linewidth=1, label="")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("frequencia [rad/s]")
ax[0].grid(True)
ax[0].set_xlim(-400, 400)
ax[0].set_title('Letra C -|Xa(e^jw)|')

ax[1].stem(wdc, PhasXc, 'r-', label="")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("frequencia [rad/s]")
ax[1].grid(True)
ax[1].set_xlim(-400, 400)
ax[1].set_title('Letra C - arg(Xa(e^jw))')

ax[2].plot(tc, xc, 'c-', linewidth=1, label="")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlabel("tempo (s)")
ax[2].grid(True)
ax[2].set_xlim(-0.5, 0.5)
ax[2].set_title('Letra C - x(t)')

#Letra D
fig, ax = plt.subplots(3,1)
# GRAFICO DO TEMPO
ax[0].stem(td, xd, 'c-', label="")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("t")
ax[0].grid(True)
ax[0].set_title('x(t)')

# MODULO FT
ax[1].plot(wdD, ModXd, 'r-',linewidth=2, label="")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("frequencia [rad/s]")
ax[1].grid(True)
ax[1].set_xlim(-1, 1)
ax[1].set_title('|Xa(e^jw)|')

# FASE FT
ax[2].stem(wdD, PhasXd, 'r-', label="")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlabel("frequencia [rad/s]")
ax[2].grid(True)
ax[2].set_xlim(-1, 1)
ax[2].set_title('arg(Xa(e^jw))')


plt.show()

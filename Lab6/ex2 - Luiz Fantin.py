
# Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift
import scipy as sc
from scipy import signal

def square(wt, duty = 0.5):
    sq = sc.signal.square(wt, duty=duty )
    return sq

# Filtro Passa-Altas
def passaAlta(freq):
    if np.abs(freq)>=1990*pi:
        return 1.0
    else:
        return 0.0



# definindo o sinal continuo periódico
fs = 1000 # freq. do sinal periódico
w0 = 2*pi*fs # frequencia angular
Ts = 1/fs # período fundamental do sinal
Tam = 1/(100*fs) # período de amostragem 100 vezes menor que o período do sinal
Np = 30 # quantidade de periodos no vetor do sinal

# Criando o vetor de tempo,5 períodos, e intervalo de Tam
t = np.arange(0,Ts*Np,Tam)

# criando x(t)
s = 2.5*square(w0*t,duty = 0.50)
frand = np.random.rand(2) # fase aleatoria para as variaveis do sinal ns(t)
Vfrq = (1+np.cos(200*t+(pi*frand[0]) ))*(100/2)
Vdc = (1+np.cos(200*t+(pi*frand[1])))*(5/2)
ns = Vdc + 2.5*np.cos(2*pi*Vfrq*t)

x = s + ns

#criando o vetor de frequencia
w = fftfreq(len(t), d=(1/Ts)*Tam)

# calculando a FS dos sinais envolvidos
X = fft(x)/len(x)

# Calculo de H[jkw0]
H = np.array([passaAlta(1j*freq*w0) for freq in w])

# calculando o modulo - magnitude do espectro
ModX = np.abs(X)
ModH = np.abs(H)

#saida
Y = H * X

# |Y[k]|
ModY = np.abs(Y)
phasY = np.angle(Y)

# y(t)
y = ifft(Y)*len(x)
y = np.real(y)


# Plot dos gráficos
fig0, ax0 = plt.subplots(3, 1)
ax0[0].plot(t, y, 'c-', lw=2,  label="y(t)")
ax0[0].set_ylabel("Amplitude")
ax0[0].set_xlabel("t")
ax0[0].grid(True)
ax0[0].set_title('y(t)')

ax0[1].stem(w, phasY, 'c-', label="angle(Y[K])")
ax0[1].set_ylabel("Amplitude")
ax0[1].set_xlabel("k")
ax0[1].grid(True)
ax0[1].set_title('angle(Y[K])')

ax0[2].stem(w, ModY, 'c-', label="|Y[K]|")
ax0[2].set_ylabel("Amplitude")
ax0[2].set_xlabel("k")
ax0[2].grid(True)
ax0[2].set_title('|Y[k]|')

fig1, ax1 = plt.subplots(3, 1)

ax1[0].stem(w, ModH, 'c-', label="|H[jkw0]|")
ax1[0].set_ylabel("Amplitude")
ax1[0].set_xlabel("k")
ax1[0].grid(True)
ax1[0].set_title('|H[jkw0]|')

ax1[1].stem(w, ModX, 'c-', label="|X[K]|")
ax1[1].set_ylabel("Amplitude")
ax1[1].set_xlabel("k")
ax1[1].grid(True)
ax1[1].set_title('|X[k]|')

ax1[2].stem(w, ModY, 'c-', label="|Y[K]|")
ax1[2].set_ylabel("Amplitude")
ax1[2].set_xlabel("k")
ax1[2].grid(True)
ax1[2].set_title('|Y[k]|')

fig0.tight_layout()
fig1.tight_layout()

plt.show()
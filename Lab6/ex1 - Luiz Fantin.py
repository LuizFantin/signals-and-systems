
# Luiz Fantin Neto - 2018102590

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from numpy import pi
from numpy.fft import fft, ifft, fftfreq, fftshift

def plot(y,x1,x2,x3):
    fig1, ax = plt.subplots(3, 1)
    ax[0].stem(y, x2, linefmt='c-', use_line_collection=True)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("k")
    ax[0].grid(True)
    ax[0].set_title('X[k]')

    ax[1].stem(y, x2, linefmt='c-', use_line_collection=True)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("k")
    ax[1].grid(True)
    ax[1].set_title('|X[k]|')

    ax[2].stem(y, x3, linefmt='c-', use_line_collection=True)
    ax[2].set_ylabel("Amplitude")
    ax[2].set_xlabel("k")
    ax[2].grid(True)
    ax[2].set_title('angle(X[k])')

    fig1.tight_layout()
    plt.show()
def plot2(y1,y2,x1,x2,x3,RC):
    fig1, ax = plt.subplots(3, 1)
    ax[0].plot(y1, x1, 'c-', lw=2, label='ns(t)')
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("k")
    ax[0].grid(True)
    ax[0].set_title('y(t) - RC ='+RC)

    ax[1].stem(y2, x2, linefmt='c-', use_line_collection=True)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("k")
    ax[1].grid(True)
    ax[1].set_title('mod[k] - RC ='+RC)

    ax[2].stem(y2, x3, linefmt='c-', use_line_collection=True)
    ax[2].set_ylabel("Amplitude")
    ax[2].set_xlabel("k")
    ax[2].grid(True)
    ax[2].set_title('angle[k] - RC ='+RC)


    fig1.tight_layout()
    plt.show()

T = 1                               
w0 = 2*pi/T                         
TAM = T/50                         
period=2                              

t=np.arange(0, T*period, TAM)         
Vs=(1/2)*signal.square(w0*t, duty=0.25) + (1/2) 

#Letra A
X = fft(Vs) / len(Vs)               
w = fftfreq(len(t), d=(1 / T)*TAM)  

Xd = fftshift(X)                    
w = fftshift(w)

ModX = np.abs(Xd)  
phasX = np.angle(Xd) 
phasX[ModX < 0.00001] = 0

plot(w,Xd,ModX,phasX)

#Letra B
RC1 = 0.01
RC2 = 0.1
RC3 = 1
j=1.j
H1 = 1/ (1 + (w0*RC1*w)*j)                
H2 = 1/ (1 + (w0*RC2*w)*j)               
H3 = 1/ (1 + (w0*RC3*w)*j)              

Yk1 = H1 * Xd                             
Yk2 = H2 * Xd                              
Yk3 = H3 * Xd                              
Yk1 = fftshift(Yk1)                       
Yk2 = fftshift(Yk2)                       
Yk3 = fftshift(Yk3) 

ModY1 = np.abs(Yk1)  
phasY1 = np.angle(Yk1)        
ModY2 = np.abs(Yk2)  
phasY2 = np.angle(Yk2)          
ModY3 = np.abs(Yk3)  
phasY3 = np.angle(Yk3)        


yr1=ifft(Yk1)*len(t)                       
yr2=ifft(Yk2)*len(t)                      
yr3=ifft(Yk3)*len(t)                    
yr1=np.real(yr1)
yr2=np.real(yr2)
yr3=np.real(yr3)

plot2(t,w,yr1,ModY1,phasY1,'0.01')
plot2(t,w,yr2,ModY2,phasY2,'0.1')
plot2(t,w,yr3,ModY3,phasY3,'1')

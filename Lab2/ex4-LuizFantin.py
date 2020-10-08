
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

def triangular(wot):
    return signal.sawtooth(wot,width = 0.5)

def main():
    
    t = np.arange(0, 0.6, 10**-3)
    x = triangular(10*pi*t)

    fig1, ax1 = plt.subplots()
    ax1.plot(t,x,'c-', linewidth = 6 ,label = "Onda triangular")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()

main()

#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

def square(wt, duty = 0.5):
    return signal.square(wt,duty)

def main():
    
    t = np.arange(0, 0.6, 10**-3)
    x = square(10*pi*t, 0.2)

    fig1, ax1 = plt.subplots()
    ax1.plot(t,x,'c-', linewidth = 6 ,label = "Onda quadrada")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()

main()
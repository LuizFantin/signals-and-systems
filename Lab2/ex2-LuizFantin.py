
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

def main():
    periodo = 2*(1/30)
    t = np.arange(0, periodo, 10**-3)
    x1 = 2* np.cos(30*pi*t)
    x2 = 2.5*np.cos(60*pi*t)+x1

    fig1, ax1 = plt.subplots()
    ax1.plot(t,x2,'c-', linewidth = 6 ,label = "Sinal Contínuo")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()

main()
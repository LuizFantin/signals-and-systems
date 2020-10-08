
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

def main():
    periodo = 2*(1/15)
    t = np.arange(0, periodo, 10**-3)
    x = 2* np.cos(30*pi*t)

    fig1, ax1 = plt.subplots()
    ax1.plot(t,x,'c-', linewidth = 6 ,label = "Sinal Cosseno Contínuo")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    print("\nValor máximo: 2\n")
    print("Valor mínimo: -2\n")

    plt.show()

main()
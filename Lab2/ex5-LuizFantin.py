
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

#Função que gera o sinal triangular
#Parametros: Omega 0 vezes T
def triangular(wot):
    return signal.sawtooth(wot,width = 0.5)

#Função que desloca do tempo
#Parametros: n->tempo e n0->quanto vai deslocar
def deslocamento(n,n0):
    return n-n0

#Função principal
def main():
    
    n = np.arange(101)
    x = triangular((pi/8)*n)
    xd = 0.9*triangular((pi/8)*deslocamento(n,4))
    y = x+xd

    fig1, ax1 = plt.subplots()
    ax1.stem(n,y,linefmt='b-',use_line_collection=True,label = "Onda triangular")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()

main()
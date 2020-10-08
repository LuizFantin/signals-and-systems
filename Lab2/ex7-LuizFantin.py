
#Luiz Fantin Neto - 2018102590

import numpy as np
from numpy import pi
import scipy as sp
import matplotlib.pyplot as plt

def deslocamento(n,n0):
    return n-n0

def degrau(n,n0):
    nd = deslocamento(n,n0)
    u = []
    for i in nd:
        if(i >= 1):
            u.append(1)
        else:
            u.append(0)
    return u

def main():
    
    n = np.linspace(-5, 10, 16)
    x1 = degrau(n,-1)
    x2 = degrau(n,0)
    x3 = degrau(n,+3)

    fig1, ax1 = plt.subplots()
    ax1.plot(n,x1,'c-', linewidth = 1 ,label = "Função degrau")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()
    fig1, ax1 = plt.subplots()
    ax1.plot(n,x2,'c-', linewidth = 1 ,label = "Função degrau")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()
    fig1, ax1 = plt.subplots()
    ax1.plot(n,x3,'c-', linewidth = 1 ,label = "Função degrau")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()
    
    plt.show()

main()
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def main():
    signal = np.array([0,0,0,2,4])
    n = np.array([-2,1,0,1,2])
    signalRev = signal[::-1]
    sp = []
    si = []

    x = 0
    while x < signal.size:
        pass
        sp.append((1/2)*(signal[x]+signalRev[x]))
        si.append((1/2)*(signal[x]-signalRev[x]))
        x+=1
    
    print("Sinal Original: ",signal)
    print("Componente Par: ",sp)
    print("Componente Ímpar: ",si)

    fig1, ax1 = plt.subplots()
    ax1.stem(signal,n,linefmt='b-',use_line_collection=True,label = "Sinal")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.stem(sp,n,linefmt='b-',use_line_collection=True,label = "Sinal")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax2.legend()
    fig2.show()

    fig3, ax3 = plt.subplots()
    ax3.stem(si,n,linefmt='b-',use_line_collection=True,label = "Sinal")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    ax1.legend()
    fig3.show()

    plt.show()

main()
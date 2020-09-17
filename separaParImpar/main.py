import numpy as np

def main():
    signal = np.array([2,1,0,1,2])
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
main()
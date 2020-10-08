import numpy as np
from numpy import pi

def main():
    print("Transformando de polar para retangular\n")
    x = float(input("Digite a amplitude do número complexo na forma polar: "))
    y = float(input("Digite a angulo do número complexo na forma polar: "))

    real = x*np.cos(np.deg2rad(y))
    imag = x*np.sin(np.deg2rad(y))

    print("\nO equivalente retangular eh:\n")
    print("real: ",real)
    print("imaginario: ",imag)
main()

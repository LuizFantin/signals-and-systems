import numpy as np
from numpy import pi

def main():
    print("Transformando de retangular para polar\n")
    real = float(input("Digite a parte real do número complexo na forma retangular: "))
    imag = float(input("Digite a parte imaginaria do número complexo na forma retangular: "))

    z = np.sqrt(real*real + imag*imag)
    theta = np.rad2deg(np.arctan(imag/real))

    print("\nO equivalente retangular eh:\n")
    print("Z: ",z)
    print("Theta: ",theta)
main()

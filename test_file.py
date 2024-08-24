import matplotlib.pyplot as plt
import numpy as np

pi = 3.14159

while True:
    fValue = input("Frequency (in Hz): ")
    try:
        fValue = float(fValue)
        break
    except ValueError:
        print("Invalid Input")

while True:
    aValue = input("Amplitude: ")
    try:
        aValue = float(aValue)
        break
    except ValueError:
        print("Invalid Input")

while True:
    phiValue = input("Phase Offset (in Radians): ")
    try:
        phiValue = float(phiValue)
        break
    except ValueError:
        print("Invalid Input")

while True:
    tValue = input("Duration (in Seconds): ")
    try:
        tValue = float(tValue)
        break
    except ValueError:
        print("Invalid Input")

while True:
    fsValue = input("Sampling Frequency (in Hz): ")
    try:
        fsValue = float(fsValue)
        break
    except ValueError:
        print("Invalid Input")

# INPUT f (float) = frequency (Hz)
# INPUT A (float) = amplitude
# INPUT phi (float) = phase offset (radians)
# INPUT t (float) = duration (seconds)
# INPUT Fs (float) = sampling frequency (Hz)
# OUTPUT x (Nx1) = sinusoidal signal

bigTValue = 1 / fsValue

x = np.arange(0, tValue, bigTValue)
y = aValue * np.cos(2 * pi * fValue * x + phiValue)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

pi = 3.14159

# Generate a complex sinusoid corresponding to frequency index k:
# INPUT k (integer) = index frequency of the DFT complex sinusoid
# INPUT N (integer) = complex sinusoid sample length
# INPUT Fs (float) = sampling frequency
# OUTPUT s (Nx1 float) = generated complex sinusoid for index k

while True:
    kValue = input("Frequency Index (int): ")
    try:
        kValue = int(kValue)
        break
    except ValueError:
        print("Invalid Input")

while True:
    bigNValue = input("Sample Length (int): ")
    try:
        bigNValue = int(bigNValue)
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

# style
mpl.style.use('seaborn')

# calculations

aValue = 1
phiValue = 0

bigTValue = 1 / fsValue

omegaValue = float(2 * pi * kValue / bigNValue)

xReal = np.arange(0, bigNValue, bigTValue)
xImag = np.arange(0, bigNValue, bigTValue)

yReal = aValue * np.cos(omegaValue * xReal + phiValue)
yImag = aValue * np.sin(omegaValue * xImag + phiValue)

# plotting

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(xReal, yReal, label='real')
ax.plot(xImag, yImag, label='imaginary')

# annotaiton

# ax.annotate('real sinusoid', xy=(xReal[0], yReal[0]),  xycoords='data',
#            xytext=((bigNValue/10), yReal[0]), textcoords='axes fraction',
#            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8)
#            )

# ax.annotate('imaginary sinusoid', xy=(xImag[0], yImag[0]),  xycoords='data',
#            xytext=((bigNValue/20), yImag[0]), textcoords='axes fraction',
#            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8)
#            )

plt.xlabel('time (seconds)')
plt.ylabel('amplitude')

plt.legend()
plt.show()

# f = 3
# t = np.arange(0,1,.01)
# phi = 0
# x = np.exp(1j*(2*np.pi*f*t + phi))
# xim = np.imag(x)
# plt.figure(1)
# plt.plot(t,np.real(x))
# plt.plot(t,xim)
# plt.axis([0,1,-1.1,1.1])
# plt.xlabel('time in seconds')
# plt.ylabel('amplitude')
# plt.show()

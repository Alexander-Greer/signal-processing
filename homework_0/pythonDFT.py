import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# style
mpl.style.use('seaborn')


def sine(xValue, frequency, amplitude = 1, offset = 0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)


fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.title('Input Signal: sin(x) + sin(8x) + sin(12x) + sin(19x) +  sin(46x)')
plt.xlabel('time (s)')
plt.ylabel('magnitude')

ax2 = fig.add_subplot(122)
plt.title('Isolated Frequencies: 1 Hz, 8 Hz, 12 Hz, 19 Hz, and 46 Hz')
plt.xlabel('frequency (Hz)')


timeDuration = 1
samplingFrequency = 200

numberOfSamples = samplingFrequency * timeDuration

domain1 = np.arange(0, timeDuration, 1.0/samplingFrequency)
range1 = sine(domain1, 1) + sine(domain1, 12) + sine(domain1, 9) + sine(domain1, 46) + sine(domain1, 8)

ax1.plot(domain1, range1)


fourier = np.fft.fft(range1)

magnitudes = []

for i in range(numberOfSamples):
    magnitudes.append(pythag(fourier.real[i], fourier.imag[i]))

dftDomain = np.arange(0, numberOfSamples)

nyquistLimit = numberOfSamples//2

nyquistDomain = np.arange(0, numberOfSamples/2)
nyquistRange = [None]*nyquistLimit

print(len(nyquistDomain))
print(len(nyquistRange))

for i in range(nyquistLimit):
    nyquistRange[i] = 2 * magnitudes[i]

nyquistAveraged = [None]*nyquistLimit

for i in range(nyquistLimit):
    nyquistAveraged[i] = nyquistRange[i] / numberOfSamples

ax2.plot(nyquistDomain, nyquistAveraged, 'go')

isolatedFrequencies = []

basicallyZero = 1.0*(10**-10)  # 1 over 10 billion

for i in range(nyquistLimit):
    if abs(nyquistAveraged[i]) > basicallyZero:
        isolatedFrequencies.append(nyquistDomain[i])

ax2.plot(nyquistDomain, nyquistAveraged, 'go')

print(isolatedFrequencies)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import math


# style
mpl.style.use('seaborn')


def sine(xValue, frequency, amplitude = 1, offset = 0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


def cosine(xValue, frequency, amplitude = 1, offset = 0):
    return amplitude * np.cos(2 * np.pi * frequency * xValue + offset)


def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)


timeDuration = 1
samplingFrequency = 200

numberOfSamples = samplingFrequency * timeDuration

fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.title('Input Signal: sin(x) + sin(8x) + sin(12x) + sin(19x) +  sin(46x)')
plt.xlabel('time (s)')
plt.ylabel('magnitude')

ax2 = fig.add_subplot(122)
plt.title('Isolated Frequencies: 1 Hz, 8 Hz, 12 Hz, 19 Hz, and 46 Hz')
plt.xlabel('frequency (Hz)')


domain1 = np.arange(0, timeDuration, 1.0/samplingFrequency)
range1 = sine(domain1, 1) + sine(domain1, 12) + sine(domain1, 9) + sine(domain1, 46) + sine(domain1, 8)

ax1.plot(domain1, range1)


# print(domain1)
# print(range1)


# sum from n = 0 to N - 1
# evaluating at n of N samples


frequencyRealBins = [0]*numberOfSamples
frequencyImagBins = [0]*numberOfSamples

for i in range(numberOfSamples):
    for n in range(numberOfSamples):
        frequencyRealBins[i] += (range1[n] * (np.cos((-2 * np.pi * i * n) / numberOfSamples)))
        frequencyImagBins[i] += (range1[n] * (np.sin((-2 * np.pi * i * n) / numberOfSamples)))


# print(frequencyRealBins)
# print(frequencyImagBins)

magnitudes = []

for i in range(numberOfSamples):
    magnitudes.append(pythag(frequencyRealBins[i], frequencyImagBins[i]))

# print(magnitudes)

dftDomain = np.arange(0, numberOfSamples)

    # ax2.plot(dftDomain, magnitudes, 'ro')


nyquistLimit = samplingFrequency//2

nyquistDomain = np.arange(0, numberOfSamples/2)
nyquistRange = [None]*nyquistLimit

for i in range(nyquistLimit):
    nyquistRange[i] = 2 * magnitudes[i]

# print(nyquistLimit)
# print(nyquistDomain)
# print(nyquistRange)

    # ax2.plot(nyquistDomain, nyquistRange, 'bo')


nyquistAveraged = [None]*nyquistLimit

for i in range(nyquistLimit):
    nyquistAveraged[i] = nyquistRange[i] / numberOfSamples

ax2.plot(nyquistDomain, nyquistAveraged, 'go')


isolatedFrequencies = []

basicallyZero = 1.0*(10**-10)  # 1 over 10 billion

for i in range(nyquistLimit):
    if abs(nyquistAveraged[i]) > basicallyZero:
        isolatedFrequencies.append(nyquistDomain[i])

print(isolatedFrequencies)

plt.show()

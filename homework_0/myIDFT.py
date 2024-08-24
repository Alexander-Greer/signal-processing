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

nyquistLimit = samplingFrequency//2

fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.title('Input Frequencies: 1 Hz, 8 Hz, 9 Hz, 12 Hz, 46 Hz')
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude')

ax2 = fig.add_subplot(122)
plt.title('Original Function')
plt.xlabel('time (s)')

inputFrequencies = [1, 8, 9, 12, 46]
inputMagnitudes = [1, 1, 1, 1, 1]
inputDomain = [0] * (nyquistLimit)
inputRange = [0] * (nyquistLimit)

if not len(inputFrequencies) == len(inputMagnitudes):
    raise ValueError('Different Number of Input Frequencies and Magnitudes')

for value in range(nyquistLimit):
    for frequencyIndex in range(len(inputFrequencies)):
        if value == inputFrequencies[frequencyIndex]:
            inputRange[value] = inputMagnitudes[frequencyIndex]
    inputDomain[value] = value

print(inputDomain)
print(inputRange)

frequnecyDomain = inputDomain
frequencyRange = inputRange

ax1.plot(frequnecyDomain, frequencyRange, 'go')

signalLength = len(frequnecyDomain)

functionDomain = np.arange(0, timeDuration, 1.0/samplingFrequency)
functionRange = []

functionReal = [0]*len(functionDomain)
functionImag = [0]*len(functionDomain)

for n in range(len(functionDomain)):
    for k in range(signalLength):
        functionReal[n] += (frequencyRange[k] * (np.cos((2 * np.pi * n * k) / signalLength)))
        functionImag[n] += (frequencyRange[k] * (np.sin((2 * np.pi * n * k) / signalLength)))

for i in range(len(functionDomain)):
    functionRange.append((1/signalLength) * (pythag(functionReal[i], functionImag[i])))

print(len(frequnecyDomain))
print(len(frequencyRange))
print(len(functionDomain))
print(len(functionRange))

ax2.plot(functionDomain, functionRange)

plt.show()

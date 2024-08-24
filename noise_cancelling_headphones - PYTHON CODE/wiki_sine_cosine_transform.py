import matplotlib.pyplot as plt
import numpy as np
import wave, struct, math, operator, scipy.fftpack

myScreenSize = (16, 7.2)


def sine(xValue, frequency, amplitude=1, offset=0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


def cosine(xValue, frequency, amplitude=1, offset=0):
    return amplitude * np.cos(2 * np.pi * frequency * xValue + offset)


def pythag(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def fourierSine(inputSignal, nOS, type):
    output = [0] * (nOS - 1)
    if type == 1:
        for frequency in range(nOS - 1):
            for value in range(nOS - 1):
                output[frequency] += inputSignal[value] * np.sin((value + 1) * (frequency + 1) * (np.pi / (nOS + 1)))
    elif type == 2:
        for frequency in range(nOS - 1):
            for value in range(nOS - 1):
                output[frequency] += inputSignal[value] * np.sin((value + 0.5) * (frequency + 1) * (np.pi / nOS))
    elif type == 3:
        for frequency in range(nOS - 1):
            output[frequency] += ((-1)**frequency)/2 * inputSignal[nOS - 1]
            for value in range(nOS - 2):
                output[frequency] += inputSignal[value] * np.sin((value + 1) * (frequency + 0.5) * (np.pi / nOS))
    elif type == 4:
        for frequency in range(nOS - 1):
            for value in range(nOS - 1):
                output[frequency] += inputSignal[value] * np.sin((value + 0.5) * (frequency + 0.5) * (np.pi / nOS))
    else:
        print("INVALID TYPE")
    return output


def fourierCosine(inputSignal, nOS, type):
    output = [0] * (nOS - 1)
    if type == 1:
        for frequency in range(nOS - 1):
            output[frequency] += 0.5 * (inputSignal[0] + (((-1) ** frequency) * inputSignal[nOS - 1]))
            for value in range(1, (nOS - 2)):
                output[frequency] += inputSignal[value] * np.cos(value * frequency * (np.pi / (nOS - 1)))
    elif type == 2:
        for frequency in range(nOS - 1):
            for value in range(nOS - 1):
                output[frequency] += inputSignal[value] * np.cos((value + 0.5) * frequency * (np.pi / nOS))
    elif type == 3:
        for frequency in range(nOS - 1):
            output[frequency] += 0.5 * inputSignal[0]
            for value in range(1, nOS - 1):
                output[frequency] += inputSignal[value] * np.cos(value * (frequency + 0.5) * (np.pi / nOS))
    elif type == 4:
        for frequency in range(nOS - 1):
            for value in range(nOS - 1):
                output[frequency] += inputSignal[value] * np.cos((value + 0.5) * (frequency + 0.5) * (np.pi / nOS))
    else:
        print("INVALID TYPE")
    return output


fig = plt.figure(figsize=myScreenSize)
plt1 = fig.add_subplot(2, 5, 1)
plt.title('Original Function')
plt2 = fig.add_subplot(2, 5, 2)
plt.title('')
plt3 = fig.add_subplot(2, 5, 3)
plt.title('')
plt4 = fig.add_subplot(2, 5, 4)
plt.title('')
plt5 = fig.add_subplot(2, 5, 5)
plt.title('')
plt7 = fig.add_subplot(2, 5, 7)
plt.title('')
plt8 = fig.add_subplot(2, 5, 8)
plt.title('')
plt9 = fig.add_subplot(2, 5, 9)
plt.title('')
plt10 = fig.add_subplot(2, 5, 10)

signalLength = 1
numberOfSamples = 500

domain = np.arange(0, signalLength, (signalLength/numberOfSamples))
inputSignal = sine(domain, 9, 1, 0)  # + sine(domain, 2, 1, 0)

# Found via trial and error
magnitudeScalar = 2 * numberOfSamples

frequencyDomain = np.arange(0, (numberOfSamples - 1))

plt1.plot(domain, inputSignal, 'b')
plt2.plot(frequencyDomain, fourierCosine(inputSignal, numberOfSamples, 1), 'ro')
plt3.plot(frequencyDomain, fourierCosine(inputSignal, numberOfSamples, 2), 'ro')
plt4.plot(frequencyDomain, fourierCosine(inputSignal, numberOfSamples, 3), 'ro')
plt5.plot(frequencyDomain, fourierCosine(inputSignal, numberOfSamples, 4), 'ro')
plt7.plot(frequencyDomain, fourierSine(inputSignal, numberOfSamples, 1), 'ko')
plt8.plot(frequencyDomain, fourierSine(inputSignal, numberOfSamples, 2), 'ko')
plt9.plot(frequencyDomain, fourierSine(inputSignal, numberOfSamples, 3), 'ko')
plt10.plot(frequencyDomain, fourierSine(inputSignal, numberOfSamples, 4), 'ko')
plt.show()

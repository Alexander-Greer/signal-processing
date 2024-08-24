import matplotlib.pyplot as plt
import numpy as np
import wave, struct, math, operator, scipy.fftpack, time

myScreenSize = (16, 7.2)


def sine(xValue, frequency, amplitude=1, offset=0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


def cosine(xValue, frequency, amplitude=1, offset=0):
    return amplitude * np.cos(2 * np.pi * frequency * xValue + offset)


def pythag(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def fourierCosine(inputFunction, nOS, duration):
    output = [0] * nOS
    for domainValue in range(nOS):
        for freq in range(nOS):
            output[domainValue] += inputFunction[domainValue] * cosine(domainValue * (duration/nOS), freq)
    return output


def fourierSine(inputFunction, nOS, duration):
    output = [0] * nOS
    for domainValue in range(nOS):
        for freq in range(nOS):
            output[domainValue] = inputFunction[domainValue] * sine(domainValue * (duration/nOS), freq)
    return output


def dctIdct(inputFunction, magScal):
    return (scipy.fftpack.dct(scipy.fftpack.idct(inputFunction, type=2), type=2))/magScal


def dstIdst(inputFunction, magScal):
    return (scipy.fftpack.dst(scipy.fftpack.idst(inputFunction, type=2), type=2))/magScal


signalLength = 1
numberOfSamples = 500

domain = np.arange(0, signalLength, (signalLength/numberOfSamples))
inputSignal = sine(domain, 9, 1, 0) + sine(domain, 2, 1, 0)

# Found via trial and error
magnitudeScalar = 2 * numberOfSamples

plt.show()

fig = plt.figure(figsize=myScreenSize)
plt1 = fig.add_subplot(221)
plt.title('Original Function')
plt1.set_ylim([(-1 * max(inputSignal) * 1.5),(max(inputSignal) * 1.5)])
plt2 = fig.add_subplot(222)
plt.title('Generated Function')
plt2.set_ylim([(-1 * max(inputSignal) * 1.5),(max(inputSignal) * 1.5)])
plt3 = fig.add_subplot(223)
plt.title('Sum')
plt3.set_ylim([(-1 * max(inputSignal) * 1.5),(max(inputSignal) * 1.5)])

plottedOriginal = [0] * numberOfSamples
plottedGenerated = [0] * numberOfSamples
cancelled = [0] * numberOfSamples

originalLine, = plt1.plot(domain, plottedOriginal, 'b')
generatedLine, = plt2.plot(domain, plottedGenerated, 'r')
cancelledLine, = plt3.plot(domain, cancelled, 'k')

for value in range(len(domain)):
    plottedOriginal[value] = inputSignal[value]
    originalLine.set_ydata(plottedOriginal)

    plottedGenerated[value] = -1 * dstIdst(inputSignal, magnitudeScalar)[value]
    generatedLine.set_ydata(plottedGenerated)

    cancelled[value] = plottedOriginal[value] + plottedGenerated[value]
    cancelledLine.set_ydata(cancelled)

    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.0005)

plt.show()



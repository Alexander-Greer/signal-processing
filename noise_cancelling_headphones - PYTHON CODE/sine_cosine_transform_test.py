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


def fourierSine(inputSignal, frequencyDomain, timeDomain):
    output = [0] * len(frequencyDomain)
    for frequency in range(len(frequencyDomain)):
        for value in range(len(timeDomain)-1):
            output[frequency] += (inputSignal[value] * sine(inputSignal[value], frequency))
    return output


def fourierCosine(inputSignal, frequencyDomain, timeDomain):
    output = [0] * len(frequencyDomain)
    for frequency in range(len(frequencyDomain)):
        for value in range(len(timeDomain)-1):
            output[frequency] += (inputSignal[value] * cosine(inputSignal[value], frequency))
    return output


fig = plt.figure(figsize=myScreenSize)
plt1 = fig.add_subplot(331)
plt.title('Original Function')
plt2 = fig.add_subplot(332)
plt.title('Cos Div')
plt3 = fig.add_subplot(333)
plt.title('Sin Div')
plt4 = fig.add_subplot(334)
plt.title('Magnitudes')
plt5 = fig.add_subplot(335)
plt.title('Cos Constructed')
plt6 = fig.add_subplot(336)
plt.title('Sin Constructed')
plt7 = fig.add_subplot(337)
plt.title('Cos Sin Sum')
plt8 = fig.add_subplot(338)
plt.title('')
plt9 = fig.add_subplot(339)
plt.title('')

signalLength = 1
numberOfSamples = 500

domain = np.arange(0, signalLength, (signalLength/numberOfSamples))
inputSignal = sine(domain, 2, 1, 0)  # + sine(domain, 2, 1, 0)

# Found via trial and error
magnitudeScalar = 2 * numberOfSamples

frequencyDomain = np.arange(0, (numberOfSamples//2))

print(domain)
print(inputSignal)

cosineTransformed = np.delete((scipy.fftpack.dct(inputSignal) / magnitudeScalar), np.s_[numberOfSamples//2:])
sineTransformed = np.delete((scipy.fftpack.dst(inputSignal) / magnitudeScalar), np.s_[numberOfSamples//2:])

magnitudes = ((sineTransformed**2) + (cosineTransformed**2))**(1/2)

cosDiv = cosineTransformed / magnitudes
sinDiv = sineTransformed / magnitudes


"""
wikiCosTransform = fourierCosine(inputSignal, numberOfSamples, signalLength)
wikiSinTransform = fourierSine(inputSignal, numberOfSamples, signalLength)

fourierInversionSum = [0] * domain

for value in range(len(domain)):
    for frequency in range(len(frequencyDomain)):
        fourierInversionSum[value] += ((wikiCosTransform[frequency] * cosine(value, frequency)) + (wikiSinTransform[frequency] * sine(value, frequency)))
"""


cosReconstructed = [0] * domain
sinReconstructed = [0] * domain

for value in range(len(domain)):
    for frequency in range(len(frequencyDomain)):
        cosReconstructed[value] += cosine(value, frequency, cosineTransformed[frequency], math.acos(cosDiv[frequency]))
        sinReconstructed[value] += sine(value, frequency, sineTransformed[frequency], math.asin(sinDiv[frequency]))

sinCosSum = cosReconstructed + sinReconstructed

notableSin = []
notableCos = []

for value in range(len(sinDiv)):
    if abs(sinDiv[value]) > 1*10**-10 and abs(sinDiv[value]) < 1:
        print("Notable SinDiv: " + str(value) + ", " + str(sinDiv[value]))
        notableSin.append(value)

for value in range(len(cosDiv)):
    if abs(cosDiv[value]) > 1*10**-10 and abs(cosDiv[value]) < 1:
        print("Notable CosDiv: " + str(value) + ", " + str(cosDiv[value]))
        notableCos.append(value)

generatedSin = [0] * domain
generatedCos = [0] * domain

notableSinValue = int(notableSin[0])
notableCosValue = int(notableCos[0])

print(notableSin[0], magnitudes[notableSinValue], sinDiv[notableSinValue])
print(notableCos[0], magnitudes[notableCosValue], cosDiv[notableCosValue])

for value in range(len(domain)):
    # for frequency in range(len(notableSin)):
    generatedSin[value] = sine(value, notableSinValue, (magnitudes[notableSinValue] * sinDiv[notableSinValue]))
    generatedCos[value] = cosine(value, notableCosValue, (magnitudes[notableCosValue] * cosDiv[notableCosValue]))

plt1.plot(domain, inputSignal, 'b')
# plt2.plot(frequencyDomain, cosDiv, 'ko')
# plt3.plot(frequencyDomain, sinDiv, 'go')
# plt4.plot(frequencyDomain, magnitudes, 'ro')
# plt5.plot(domain, cosReconstructed, 'k')
# plt6.plot(domain, sinReconstructed, 'g')
# plt7.plot(domain, sinCosSum, 'c')
# plt8.plot(domain, generatedCos, 'm')
# plt9.plot(domain, generatedSin, 'y')

plt2.plot(frequencyDomain, fourierSine(inputSignal, frequencyDomain, domain), 'r')
plt3.plot(frequencyDomain, fourierCosine(inputSignal, frequencyDomain, domain), 'k')
plt.show()

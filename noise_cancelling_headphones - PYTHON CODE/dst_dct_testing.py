import matplotlib.pyplot as plt
import numpy as np
import wave, struct, math
import operator
import scipy.fftpack

myScreenSize = (16, 7.2)


def sine(xValue, frequency, amplitude=1, offset=0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


def pythag(x, y):
    return math.sqrt(x ** 2 + y ** 2)


# "A440.wav"
# "440_600_40960.wav"
# "Sheep_Bleat.wav"
# "250Hz.wav"
wavFile = "440_600_40960.wav"

# https://stackoverflow.com/ questions/2060628/reading-wav-files-in-python
file = wave.open(wavFile, 'rb')

# get data
# https://docs.python.org/3/library/wave.html

channel = file.getnchannels()
sampleWidth = file.getsampwidth()
Fs = file.getframerate()
numberOfSamples = file.getnframes()
compressionType = file.getcomptype()
compressionName = file.getcompname()
parameters = file.getparams()
duration = numberOfSamples / Fs
bitNumber = 8 * sampleWidth

frequencyResolution = Fs / numberOfSamples
print(frequencyResolution)

print(file.getparams())

nyquistDomain = np.arange(0, Fs // 2, frequencyResolution)
sineAveraged = [0] * len(nyquistDomain)
cosineAveraged = [0] * len(nyquistDomain)

amplitudes = [0] * len(nyquistDomain)
differenceRatios = [0] * len(nyquistDomain)


# https://www.youtube.com/watch?v=mkGsMWi_j4Q
def fourierDistribute(inputSignal):
    # perform fourier transform on input data
    fourieredSine = scipy.fftpack.dst(inputSignal)
    fourieredCosine = scipy.fftpack.dct(inputSignal)

    # iterate through every data point within the nyquist limit (half the sampling frequency)
    for number in range(int(Fs // (2 * frequencyResolution))):
        # spread the data out along the the frequency plot at intervals equal to the frequency resolution
        # (the data is first converted from its imaginary vector form into its magnitude...
        # the magnitude doubled because of the nyquist transformation...
        # and then averaged out across the number of samples taken)
        sineAveraged[number] = pythag(fourieredSine.real[number], fourieredSine.imag[number]) * 2 // numberOfSamples
        cosineAveraged[number] = pythag(fourieredCosine.real[number], fourieredCosine.imag[number]) * 2 // numberOfSamples
        amplitudes[number] = sineAveraged[number] + cosineAveraged[number]
        if (sineAveraged[number] != 0) and (cosineAveraged != 0):
            differenceRatios[number] = sineAveraged[number] // cosineAveraged[number]

    # # https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list/34889013
    # maxFreq, maxValue = max(enumerate(nyquistAveraged), key=operator.itemgetter(1))
    # print(maxFreq, maxValue)

    # return the data corrected for the nyquist limit and frequency resolution
    return nyquistDomain, sineAveraged, cosineAveraged, amplitudes, differenceRatios


# https://stackoverflow.com/questions/3957025/what-does-a-audio-frame-contain
resolution = str(sampleWidth * 8) + '-bit'

signalType = 'mono' if channel == 1 else 'stereo'

bufferBitSize = '<h' if signalType == 'mono' else '<i'

print(resolution + ' ' + signalType)

# begin plotting (2x2 grid of plots)

fig = plt.figure(figsize=myScreenSize)

tlplot = fig.add_subplot(231)
plt.title('WAV File')
plt.xlabel('time (s)')
plt.ylabel('magnitude')

blplot = fig.add_subplot(233)
plt.title('Spectrogram')
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')

trplot = fig.add_subplot(232)
plt.title('Whole DFT')
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude')

brplot = fig.add_subplot(234)
plt.title('Zoomed DFT')
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude')

# organize "frames"
# https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
frames = []

for i in range(0, numberOfSamples):
    waveData = file.readframes(1)
    # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    data = struct.unpack(bufferBitSize, waveData)
    frames.append(int(data[0]))

# frames = np.cos(2 * np.pi * 12 * np.arange(0, duration, 1/Fs))

fourierX = fourierDistribute(frames)[0]
fourierY = fourierDistribute(frames)[1]
fourierMaxFreq = fourierDistribute(frames)[2]
fourierMaxValue = fourierDistribute(frames)[3]

print(len(fourierX))
print(len(fourierY))

frequencyZoomX = []
frequencyZoomY = []

for sample in range(len(fourierX)):
    if fourierY[sample] >= (fourierMaxValue // 10):
        frequencyZoomX.append(fourierX[sample])
        frequencyZoomY.append(fourierY[sample])

# plot
tlplot.plot(np.arange(0, duration, 1 / Fs), frames)
blplot.specgram(frames, Fs=Fs)
trplot.plot(fourierX, fourierY, 'r-')
brplot.plot(frequencyZoomX, frequencyZoomY, 'go')

# https://matplotlib.org/users/tight_layout_guide.html
plt.tight_layout()
plt.show()

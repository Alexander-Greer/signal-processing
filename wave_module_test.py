import matplotlib.pyplot as plt
import numpy as np
import wave, struct, math
import operator

myScreenSize = (16,7.2)


def sine(xValue, frequency, amplitude = 1, offset = 0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)

def pythag(x, y):
    return math.sqrt(x ** 2 + y ** 2)

# "A440.wav"
# "440_600_40960.wav"
# "Sheep_Bleat.wav"
# "250Hz.wav"
wavFile = "Sheep_Bleat.wav"

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
nyquistAveraged = [0] * len(nyquistDomain)

# https://www.youtube.com/watch?v=mkGsMWi_j4Q
def fourierDistribute(inputSignal):
    # perform fourier transform on input data
    fouriered = np.fft.fft(inputSignal)

    # iterate through every data point within the nyquist limit (half the sampling frequency)
    for number in range(int(Fs // (2 * frequencyResolution))):
        # spread the data out along the the frequency plot at intervals equal to the frequency resolution
        # (the data is first converted from its imaginary vector form into its magnitude...
        # the magnitude doubled because of the nyquist transformation...
        # and then averaged out across the number of samples taken)
        nyquistAveraged[number] = pythag(fouriered.real[number], fouriered.imag[number]) * 2 // numberOfSamples

    # https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list/34889013
    maxFreq, maxValue = max(enumerate(nyquistAveraged), key=operator.itemgetter(1))
    print(maxFreq, maxValue)

    # return the data corrected for the nyquist limit and frequency resolution
    return nyquistDomain, nyquistAveraged, maxFreq, maxValue

"""
def nyquistFourier(inputSignal, nOS=numberOfSamples, samplingFrequency=Fs):
    fourier = np.fft.fft(inputSignal)

    magnitudes = []

    for sample in range(nOS):
        magnitudes.append(pythag(fourier.real[sample], fourier.imag[sample]))

    # dftDomain = np.arange(0, nOS)

    nyquistLimit = nOS // 2

    nyquistDomain = np.arange(0, nOS / 2)
    nyquistRange = [None] * nyquistLimit

    # print(len(nyquistDomain))
    # print(len(nyquistRange))

    for sample in range(nyquistLimit):
        nyquistRange[sample] = 2 * magnitudes[sample]

    nyquistAveraged = [None] * nyquistLimit

    for sample in range(nyquistLimit):
        nyquistAveraged[sample] = nyquistRange[sample] / nOS

    # ax2.plot(nyquistDomain, nyquistAveraged, 'go')


    isolatedFrequencies = []

    basicallyZero = 1.0 * (10 ** -10)  # 1 over 10 billion

    for sample in range(nyquistLimit):
        if abs(nyquistAveraged[sample]) > basicallyZero:
            isolatedFrequencies.append(nyquistDomain[sample])
            
    maxFreq, maxValue = max(enumerate(nyquistAveraged), key=operator.itemgetter(1))
    print(maxFreq, maxValue)

    return nyquistDomain, nyquistAveraged, maxFreq, maxValue
"""


# https://stackoverflow.com/questions/3957025/what-does-a-audio-frame-contain
resolution = str(sampleWidth * 8) + '-bit'

signalType = 'mono' if channel == 1 else 'stereo'

bufferBitSize = '<h' if signalType == 'mono' else '<i'

print(resolution + ' ' + signalType)


# begin plotting (2x2 grid of plots)

fig = plt.figure(figsize=myScreenSize)

tlplot = fig.add_subplot(221)
plt.title('WAV File')
plt.xlabel('time (s)')
plt.ylabel('magnitude')

blplot = fig.add_subplot(223)
plt.title('Spectrogram')
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')

trplot = fig.add_subplot(222)
plt.title('Whole DFT')
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude')

brplot = fig.add_subplot(224)
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


"""
# perform transform (stackoverflow)

# https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
b = [(ele/2**bitNumber)*2-1 for ele in frames] # track bit-number, b is now normalized on [-1,1)
c = np.fft.fft(b)  # calculate fourier transform (complex numbers list)
d = len(c)//2  # you only need half of the fft list (real signal symmetry)
e = abs(c[:(d-1)])  # get the positive frequency values
e = e[20:20000]
# index, value = max(enumerate(e), key=operator.itemgetter(1))
# print(index, value)
"""


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

print(fourierX)
print("DIVIDER")
print(fourierY)

# plot
tlplot.plot(np.arange(0, duration, 1/Fs), frames)
blplot.specgram(frames, Fs=Fs)
trplot.plot(fourierX, fourierY, 'r-')
brplot.plot(frequencyZoomX, frequencyZoomY, 'go')

# https://matplotlib.org/users/tight_layout_guide.html
plt.tight_layout()
plt.show()

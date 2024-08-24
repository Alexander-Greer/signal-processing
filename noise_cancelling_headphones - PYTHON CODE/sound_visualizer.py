#!usr/bin/env python
#coding=utf-8
# https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python

import matplotlib.pyplot as plt
import numpy as np
import struct, math, time
import operator

import pyaudio
import wave

def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)

myScreenSize = (16, 7.2)

#define stream chunk
chunk = 4096

# "Chopin_Nocturne_Opus_9_2.WAV"
# "A440.wav"
# "440_600_40960.wav"
# "Sheep_Bleat.wav"
# "250Hz.wav"
# "piano_C_major.wav"
filename = "Chopin_Nocturne_Opus_9_2.WAV"


# open a wav format music
# https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
file = wave.open(filename, "rb")

# instantiate PyAudio
# https://people.csail.mit.edu/hubert/pyaudio/docs/
p = pyaudio.PyAudio()
# open stream
stream = p.open(format = p.get_format_from_width(file.getsampwidth()),
                channels = file.getnchannels(),
                rate = file.getframerate(),
                output = True)

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

frequencyResolution = Fs // chunk
# print(frequencyResolution)


pianoRange = (28, 4186)
hearingRange = (20, 20000)
instRange = pianoRange
convertedRange = (instRange[0] // frequencyResolution, instRange[1] // frequencyResolution)


domain = np.arange(0, Fs // 2, frequencyResolution)
frequencyX = domain[convertedRange[0]:convertedRange[1]]
frequencyY = [None] * len(frequencyX)

# https://www.youtube.com/watch?v=mkGsMWi_j4Q
def fourierDistribute(inputSignal):
    # perform fourier transform on input data
    fouriered = np.fft.fft(inputSignal)

    # iterate through every data point within the nyquist limit (half the sampling frequency)
    for number in range(len(frequencyX)):
        # spread the data out along the the frequency plot at intervals equal to the frequency resolution
        # (the data is first converted from its imaginary vector form into its magnitude...
        # the magnitude doubled because of the nyquist transformation...
        # and then averaged out across the number of samples taken)
        frequencyY[number] = pythag(fouriered.real[number], fouriered.imag[number]) * 2 // chunk

    # return the data corrected for the nyquist limit and frequency resolution
    return frequencyY


# https://stackoverflow.com/questions/3957025/what-does-a-audio-frame-contain
resolution = str(sampleWidth * 8) + '-bit'
signalType = 'mono' if channel == 1 else 'stereo'
audioType = resolution + ' ' + signalType
print(audioType)

bitSize = '<' + str(chunk) + ('H' if signalType == 'mono' else 'L')
print(bitSize)

plt.show()

myScreenSize = (16/2,7.2/2)
fig = plt.figure(figsize=myScreenSize)

plot = fig.add_subplot(111)
plt.title(filename)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

ax = plt.gca()
ax.set_xscale('linear')

# read data
data = file.readframes(chunk)

print(len(data))

values = []

# https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
firstChunk = struct.unpack(bitSize, data)

values = (list(firstChunk))

line1, = plot.plot(frequencyX, fourierDistribute(values), 'r-')

# print(len(frequencyX))
# print(len(fourierDistribute(values)))

# play stream
while data:
    stream.write(data)
    data = file.readframes(chunk)

    # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    firstChunk = struct.unpack(bitSize, data)
    values = (list(firstChunk))

    line1.set_xdata(frequencyX)
    line1.set_ydata(fourierDistribute(values))

    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.0005)


#stop stream
stream.stop_stream()
stream.close()

#close PyAudio
p.terminate()

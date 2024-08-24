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

# https://gist.github.com/mabdrabo/8678538
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
# https://stackoverflow.com/questions/6560680/pyaudio-memory-error
CHUNK = 4096
RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"

#instantiate PyAudio
# https://people.csail.mit.edu/hubert/pyaudio/docs/
audio = pyaudio.PyAudio()
#open stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("recording...")

# get data
# https://docs.python.org/3/library/wave.html

channel = CHANNELS
sampleWidth = 2
Fs = RATE

# numberOfSamples = file.getnframes()
# compressionType = file.getcomptype()
# compressionName = file.getcompname()
# parameters = file.getparams()
# duration = numberOfSamples / Fs

bitNumber = 8 * sampleWidth

frequencyResolution = Fs // CHUNK
print(frequencyResolution)

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
        frequencyY[number] = pythag(fouriered.real[number], fouriered.imag[number]) * 2 // CHUNK

    # return the data corrected for the nyquist limit and frequency resolution
    return frequencyY

plt.show()

myScreenSize = (16/2,7.2/2)
fig = plt.figure(figsize=myScreenSize)

plot = fig.add_subplot(111)
plt.title('Microphone Input')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

ax = plt.gca()
ax.set_xscale('linear')


values = []

#read data
data = stream.read(CHUNK)

# https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
firstChunk = struct.unpack('<4096L', data)

# https://stackoverflow.com/questions/3288250/how-do-i-get-integers-from-a-tuple-in-python/3288270
values = (list(firstChunk))

line1, = plot.plot(frequencyX, fourierDistribute(values), 'r-')

print(len(frequencyX))
print(len(fourierDistribute(values)))

#analyze stream
while data:
    data = stream.read(CHUNK)

    # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    firstChunk = struct.unpack('<4096L', data)
    values = (list(firstChunk))

    line1.set_xdata(frequencyX)
    line1.set_ydata(fourierDistribute(values))

    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.0005)

print("finished recording")

#stop stream
stream.stop_stream()
stream.close()

#close PyAudio
audio.terminate()

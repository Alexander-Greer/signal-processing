#!usr/bin/env python
#coding=utf-8
# https://stackoverflow.com/questions/17657103/how-to-play-wav-file-in-python

import matplotlib.pyplot as plt
import numpy as np
import struct, math, time
import operator
import peakutils

import pyaudio
import wave

# https://www.johndcook.com/blog/2016/02/10/musical-pitch-notation/
from math import log2, pow

A4 = 440
C0 = A4 * pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch(freq):
    # For a pitch P, the number of half steps from C0 to P is:
    h = round(12 * log2(freq / C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)


def cutoff(amplitudes):
    output = list(map(int, frequencyMultiplier * peakutils.indexes(amplitudes, thres=0.05, min_dist=1, thres_abs=False)))
    return output
    """buffer = amplitudes[30:]
    for iteration in range(60):
        buffer[iteration] = 0
    for iteration in range(60):
        print(len(buffer) - iteration)
        buffer[len(buffer) - iteration] = 0
    output = []

    for time in range(3):
        output.append(frequencyX[buffer.index(max(buffer))])

        for iteration in range(100):
            buffer[(output[time]-50) + iteration] = 0

    return output
    """


def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)

# https://gist.github.com/mabdrabo/8678538
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
# https://stackoverflow.com/questions/6560680/pyaudio-memory-error
CHUNK = 8192
RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"

#instantiate PyAudio
# https://people.csail.mit.edu/hubert/pyaudio/docs/
audio = pyaudio.PyAudio()
#open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

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
frequencyMultiplier = Fs / CHUNK
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
firstChunk = struct.unpack('<8192L', data)

# https://stackoverflow.com/questions/3288250/how-do-i-get-integers-from-a-tuple-in-python/3288270
values = (list(firstChunk))

line1, = plot.plot(frequencyX, fourierDistribute(values), 'r-')

print(len(frequencyX))
print(len(fourierDistribute(values)))

#analyze stream
while data:
    data = stream.read(CHUNK)

    # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    firstChunk = struct.unpack('<8192L', data)
    values = (list(firstChunk))

    fouriered = fourierDistribute(values)

    line1.set_xdata(frequencyX)
    line1.set_ydata(fouriered)
    print(cutoff(fouriered))

    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.0005)

print("finished recording")

#stop stream
stream.stop_stream()
stream.close()

#close PyAudio
audio.terminate()

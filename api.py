import flask
from flask import request, jsonify
from flask_cors import CORS

import pyaudio, struct, math, peakutils
import numpy as np
from math import log2, pow

# https://gist.github.com/mabdrabo/8678538
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
# https://stackoverflow.com/questions/6560680/pyaudio-memory-error
CHUNK = 4096
RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"

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

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

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


# https://stackoverflow.com/questions/3957025/what-does-a-audio-frame-contain
resolution = str(sampleWidth * 8) + '-bit'
signalType = 'mono' if channel == 1 else 'stereo'
audioType = resolution + ' ' + signalType
print(audioType)

bitSize = '<' + str(CHUNK) + ('H' if signalType == 'mono' else 'L')

def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)

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

@app.route('/', methods=['GET'])
def home():
    return '''<h1>dAnK FoRdieR TrabSfoRm ApI</p>'''


def transform(input_data):

    data = input_data.read(CHUNK)

    # TODO: DETERMINE CHUNK SIZE
    firstChunk = struct.unpack(bitSize, data)

    values = list(firstChunk)

    while data:
        data = input_data.read(CHUNK)

        # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
        firstChunk = struct.unpack(bitSize, data)
        values.append(list(firstChunk))



# Send the API the original audio file and process the file
@app.route('/api/send/', methods=['POST'])
def api_process():
    input = request.form



app.run()

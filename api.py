import flask
from flask import request, jsonify
from flask_cors import CORS

import struct, math, peakutils, wave
import numpy as np
from math import log2, pow

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

# TO BE IMPLEMENTED \/
"""
def cutoff(amplitudes):
    output = list(map(int, frequencyMultiplier * peakutils.indexes(amplitudes, thres=0.05, min_dist=1, thres_abs=False)))
    return output
"""

def pythag(xValue, yValue):
    return math.sqrt(xValue ** 2 + yValue ** 2)



@app.route('/', methods=['GET'])
def home():
    return '''<h1>dAnK FoRdieR TrabSfoRm ApI</h1>'''


def transform(input_data):
    input_file = open("input_file.wav", 'wb')
    input_file.write(input_data)
    input_file.close()

    file = wave.open("input_file", 'rb')

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
    frequencyMultiplier = Fs / numberOfSamples

    pianoRange = (28, 4186)
    hearingRange = (20, 20000)
    instRange = pianoRange
    convertedRange = (instRange[0] // frequencyResolution, instRange[1] // frequencyResolution)

    domain = np.arange(0, Fs // 2, frequencyResolution)
    frequencyX = domain[convertedRange[0]:convertedRange[1]]
    frequencyY = [None] * len(frequencyX)

    # https://stackoverflow.com/questions/3957025/what-does-a-audio-frame-contain
    resolution = str(sampleWidth * 8) + '-bit'
    signalType = 'mono' if channel == 1 else 'stereo'
    audioType = resolution + ' ' + signalType
    print(audioType)

    bitSize = '<' + str(numberOfSamples) + ('H' if signalType == 'mono' else 'L')

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
            frequencyY[number] = pythag(fouriered.real[number], fouriered.imag[number]) * 2 // numberOfSamples

        # return the data corrected for the nyquist limit and frequency resolution
        return frequencyY

    frames = []

    for i in range(0, numberOfSamples):
        waveData = file.readframes(1)
        # https://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
        data = struct.unpack(bitSize, waveData)
        frames.append(int(data[0]))

    fourierX = fourierDistribute(frames)[0]
    fourierY = fourierDistribute(frames)[1]

    return fourierX, fourierY


# Send the API the original audio file and process the file
@app.route('/api/send/', methods=['POST'])
def api_process():
    input_data = request.get_data()
    print(str(input_data))
    return jsonify(transform(input_data))


app.run()

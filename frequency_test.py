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


while True:
    frequency = input("Frequency: ")
    try:
        frequency = int(frequency)
        print(pitch(frequency))
    except ValueError:
        print("Invalid Input")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ----- Sinusoid Generator -----

""" Fill in the TODOs to plot a graph of a sinusoid! """

# ------------------------------

# Graph styling and configuration
mpl.style.use('seaborn')  # (Makes the graph easily readable)

fig = plt.figure()
ax = fig.add_subplot(111)  # (creates a singular graph to be displayed)

plt.xlabel('Time (seconds)')  # (Labels the x-axis)
plt.ylabel('Amplitude')  # (Labels the y-axis)

# Gets the frequency of the sinusoid
while True:
    frequency = input("Frequency (in Hz): ")
    try:
        frequency = float(frequency)
        break
    except ValueError:
        print("Invalid Input")

# Gets the amplitude of the sinusoid
while True:
    amplitude = input("Amplitude: ")
    try:
        amplitude = float(amplitude)
        break
    except ValueError:
        print("Invalid Input")

# Gets the phase offset of the sinusoid
while True:
    phaseOffset = input("Phase Offset (in Radians): ")
    try:
        phaseOffset = float(phaseOffset)
        break
    except ValueError:
        print("Invalid Input")

# Gets the duration of the sinusoid
while True:
    duration = input("Duration (in Seconds): ")
    try:
        duration = float(duration)
        break
    except ValueError:
        print("Invalid Input")

# Gets the sampling frequency of the sinusoid
while True:
    samplingFrequency = input("Sampling Frequency (in Hz): ")
    try:
        samplingFrequency = float(samplingFrequency)
        break
    except ValueError:
        print("Invalid Input")


# calculate sampling period based on the inputted sampling frequency
samplingPeriod = 1 / samplingFrequency

""" 
----- TODO -----

The function `np.arange()` creates a list of values
based on three parameters: (start, stop, step)
start is where the range begins (the minimum),
stop is where the range ends (the maximum),
and step is how much space in between each value.

This function is used to create the x-values of your
sinusoid. Knowing this, fill in the values for
sinusoidLength and timeBetweenSamples, each using
one of the variables inputted/calculated above.

"""

# sinusoidLength = ???
sinusoidLength = 1

# timeBetweenSamples = ???
timeBetweenSamples = 10

x = np.arange(0, sinusoidLength, timeBetweenSamples)

# ------------

"""
----- TODO -----

The function `np.cos()` creates a list of values
corresponding to the cosine of a set of numbers.

This function is used to create the y-values of your
sinusoid. Knowing this, fill in the values for SOMETHING,
SOMETHING_ELSE, and ANOTHER_SOMETHING based on your
knowledge of the equation of a sinusoid, each using
one of the variables inputted/calculated above.

Hint: The Equation of a sinusoid is:

           s[n] = A * cos(2pi * f * n * T + phi)

(You don't need to worry about T (phase period) in this function
 as it should already have been incorporated into a previous function)

"""

# SOMETHING = ???
SOMETHING = 1

# SOMETHING_ELSE = ???
SOMETHING_ELSE = 1

# ANOTHER_SOMETHING = ???
ANOTHER_SOMETHING = 0

y = SOMETHING * np.cos(2 * np.pi * SOMETHING_ELSE * x + ANOTHER_SOMETHING)

# ------------

# Plots the graph of the sinusoid and displays it to the user
ax.plot(x, y)
plt.show()

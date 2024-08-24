import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ----- Complex Sinusoid Generator -----

""" Fill in the TODOs to plot a graph of a complex sinusoid! """

# ------------------------------

# Graph styling and configuration
mpl.style.use('seaborn')  # (Makes the graph easily readable)

fig = plt.figure()
ax = fig.add_subplot(111)  # (creates a singular graph to be displayed)

plt.xlabel('Time (seconds)')  # (Labels the x-axis)
plt.ylabel('Amplitude')  # (Labels the y-axis)

# Gets the frequency index of the sinusoid
while True:
    frequencyIndex = input("Frequency Index (int): ")
    try:
        frequencyIndex = int(frequencyIndex)
        break
    except ValueError:
        print("Invalid Input")

# Gets the sample length of the complex sinusoid
while True:
    sampleLength = input("Sample Length (int): ")
    try:
        sampleLength = int(sampleLength)
        break
    except ValueError:
        print("Invalid Input")

# Gets the sampling frequency of the complex sinusoid
while True:
    samplingFrequency = input("Sampling Frequency (in Hz): ")
    try:
        samplingFrequency = float(samplingFrequency)
        break
    except ValueError:
        print("Invalid Input")


# initialize amplitude and phase offset to
# prevent unnecessarily complicated functions
amplitude = 1
phaseOffset = 0

# calculate sampling period based on the inputted sampling frequency
samplingPeriod = 1 / samplingFrequency

""" 
----- TODO -----

The following variable `omega` is used to describe
frequency in radians rather than in Hz.

Knowing what you do about this conversion, fill
in the values of SOMETHING_1 and SOMETHING_2 using
variables inputted/calculated above to properly 
convert the frequencies.

Hint: Index Frequency divided by Sample Length 
is equal to traditional frequency.

"""

# SOMETHING_1 = ???
SOMETHING_1 = 1

# SOMETHING_2 = ???
SOMETHING_2 = 1

omega = float(2 * np.pi * (SOMETHING_1 / SOMETHING_2))

# ------------

# Creates the x-values of the complex sinusoid
# based on the inputted sample length and sampling period
x = np.arange(0, sampleLength, samplingPeriod)

# ------------

""" 
----- TODO -----

The function `np.exp()` calculates the
exponential values of "e" raised to the
provided list of values. Python recognizes
"j" as the imaginary number in this case,
and so it can be used to graph your
complex sinusoid.

The following function uses the exponential
function to create the y-values for your 
complex sinusoid. Remember that a
complex sinusoid can be simplified using
Euler's Formula:

    e^(ix) = cos(x) + i * sin(x)

Knowing this, fill in the values for
SOMETHING_3 and SOMETHING_4 using
variables inputted/calculated above
in order to complete the expression 
that graphs your complex sinusoid.

"""

# SOMETHING_3 = ???
SOMETHING_3 = 0

# SOMETHING_4 = ???
SOMETHING_4 = 0

y = amplitude * np.exp(1j * ((SOMETHING_3 * x * SOMETHING_4) + phaseOffset))

# ------------

# Plots the real and imaginary portions of the complex sinusoid
ax.plot(x, y.real, label='real')
ax.plot(x, y.imag, label='imaginary')

# Displays the graph to the user
plt.legend()
plt.show()

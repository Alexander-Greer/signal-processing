import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# style
mpl.style.use('seaborn')


def sine(xValue, frequency, amplitude = 1, offset = 0):
    return amplitude * np.sin(2 * np.pi * frequency * xValue + offset)


timeDuration = 1
samplingFrequency = 200

fig = plt.figure()

ax1 = fig.add_subplot(515)
plt.ylabel('1Hz')
plt.xlabel('time (s)')

ax2 = fig.add_subplot(514)
plt.ylabel('3 Hz')

ax3 = fig.add_subplot(513)
plt.ylabel('4 Hz')

ax4 = fig.add_subplot(512)
plt.ylabel('9 Hz')

ax5 = fig.add_subplot(511)
plt.ylabel('Composite Function')

domain = np.arange(0, timeDuration, 1.0/samplingFrequency)
range1 = sine(domain, 1)
range2 = sine(domain, 3)
range3 = sine(domain, 4)
range4 = sine(domain, 9)
range5 = range1 + range2 + range3 + range4

ax1.plot(domain, range1, 'r')
ax2.plot(domain, range2, 'g')
ax3.plot(domain, range3, 'm')
ax4.plot(domain, range4, 'y')
ax5.plot(domain, range5, 'c')

plt.show()

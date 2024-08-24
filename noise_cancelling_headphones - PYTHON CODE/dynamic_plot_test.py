"""
import matplotlib as plot
import numpy as np

myScreenSize = (16,7.2)

fig = plt.figure(figsize=myScreenSize)
plt.subplot(211)

tplot = fig.add_subplot(211)
bplot = fig.add_subplot(212)

def update_plot(plot, data):
    plot.set_xdata(np.append(plot.get_xdata(), data))
    plot.set_ydata(np.append(plot.get_ydata(), data))
    plt.draw()

update_plot()
"""

import matplotlib.pyplot as plt
import time
import numpy as np

duration = 1
samplingFrequency = 500

frequency = 10
amplitude = 1

numberOfSamples = int(duration * samplingFrequency)

speed = 20

xdata = np.arange(0, duration, 1/samplingFrequency)
ydata = (amplitude * np.cos(2 * np.pi * frequency * xdata))

plt.show()

myScreenSize = (16,7.2)

fig = plt.figure(figsize=myScreenSize)
plt.subplot(211)

tplot = fig.add_subplot(211)
bplot = fig.add_subplot(212)

#axes = plt.gca()
#axes.set_xlim(0, 100)
#axes.set_ylim(-50, +50)
line1, = tplot.plot(xdata, ydata, 'r-')

for i in range(int(numberOfSamples)):
    #xdata.append(i)
    #ydata.append(np.sin(i))
    line1.set_xdata(xdata[0:i])
    line1.set_ydata(ydata[0:i])
    plt.draw()
    plt.pause(1e-17)
    time.sleep((1/speed) * (1/numberOfSamples))

bplot.specgram(ydata, NFFT=(samplingFrequency-1), Fs=samplingFrequency, noverlap=0)

# add this if you don't want the window to disappear at the end
plt.show()



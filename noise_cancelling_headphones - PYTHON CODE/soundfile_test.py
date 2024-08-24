# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile

# sample_rate, samples = wavfile.read('440_600_40960.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# read audio samples
input_data = read("A440.wav")
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:1024])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Sample Wav")
# display the plot
plt.show()

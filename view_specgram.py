import glob
import sys
import os

from functools import partial

from scipy.io import wavfile
from matplotlib import pyplot as plt

class SpecgramViewer(object):
    def __init__(self, filenames):
        self.filenames = filenames
        self.file_index = 0
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', partial(self.key_press))
        self.plot_current_call()
        plt.show()

    def key_press(self, event):
        if event.key in ('n', 'right'):
            self.file_index = (self.file_index + 1) % len(self.filenames)
            self.plot_current_call()
        elif event.key in ('p', 'left'):
            self.file_index = (self.file_index - 1) % len(self.filenames)
            self.plot_current_call()
        elif event.key in ('q', 'escape'):
            plt.close()

    def plot_current_call(self):
        self.fig.clear()

        filename = self.filenames[self.file_index]
        fs, data = wavfile.read(filename)
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)

        for ch in range(data.shape[1]):
            ax = self.fig.add_subplot(data.shape[1], 1, ch + 1)
            ax.set_title('%s, channel %i' % (filename, ch + 1))
            # TODO: edit spectrogram parameter
            # SEE: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.specgram
            ax.specgram(data[:, ch], Fs=fs)
        self.fig.canvas.draw()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: %s path_to_call_folder' % sys.argv[0])
        exit()

    # Change current working directory
    os.chdir(sys.argv[1])
    # Retrieve filenames
    filenames = glob.glob('*.wav')
    viewer = SpecgramViewer(filenames)

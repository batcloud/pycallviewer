import glob
import sys
import os

import summary

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
        self.summarizer = summary.Summarizer()
        plt.show()

    def key_press(self, event):
        if event.key in ('n', 'right'):
            self.file_index = (self.file_index + 1) % len(self.filenames)
            self.plot_current_call()
        elif event.key in ('p', 'left'):
            self.file_index = (self.file_index - 1) % len(self.filenames)
            self.plot_current_call()
        elif event.key in ('down',):
            print("".join((chr(27),"[2J",chr(27),"[H")))
            print(self.summarizer.summarize(self.data, self.fs))
        elif event.key in ('q', 'escape'):
            plt.close()

    def plot_current_call(self):
        self.fig.clear()

        filename = self.filenames[self.file_index]
        try:
            self.fs, self.data = wavfile.read(filename)
        except ValueError:
            print('File %s cannot be read as a wav file.' % filename)
            return
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(self.data.shape[0], 1)

        for ch in range(self.data.shape[1]):
            ax = self.fig.add_subplot(self.data.shape[1], 1, ch + 1)
            ax.set_title('%s, channel %i' % (filename, ch + 1))
            # TODO: edit spectrogram parameter
            # SEE: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.specgram
            ax.specgram(self.data[:, ch], Fs=self.fs)
        self.fig.canvas.draw()

import re
numbers = re.compile(r'(\d+)')
def sort_filename(value):
    parts = numbers.split(value)
    parts[0] = parts[0].lower()
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: %s path_to_call_folder' % sys.argv[0])
        exit()

    # Change current working directory
    os.chdir(sys.argv[1])
    # Retrieve filenames
    filenames = sorted(glob.glob('*.wav'), key=sort_filename)
    viewer = SpecgramViewer(filenames)

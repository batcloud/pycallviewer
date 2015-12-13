import glob
import sys
import os

from scipy.io import wavfile
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print('Usage: %s path_to_call_folder' % sys.argv[0])
    exit()

# Change current working directory
os.chdir(sys.argv[1])
# Retrieve filenames
filenames = glob.glob('*.wav')
file_index = 0

# Call visualization loop
input_ = ''
while input_ != 'q':
    file_ = filenames[file_index]
    fs, data = wavfile.read(file_)
    if len(data.shape) == 1:
        data = data.reshape(data.shape[0], 1)

    for ch in range(data.shape[1]):
        plt.subplot('%i1%i' % (data.shape[1], ch + 1))
        plt.title('%s, channel %i' % (file_, ch + 1))
        # TODO: edit spectrogram parameter
        # SEE: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.specgram
        plt.specgram(data[:, ch], Fs=fs)

    plt.show(block=False)
    input_ = ''
    while not input_ in ('p', 'n', 'q'):
        input_ = input('next (n), previous (p), quit (q): ')
    if input_ == 'n':
        file_index = (file_index + 1) % len(filenames)
    if input_ == 'p':
        file_index = (file_index - 1) % len(filenames)

    plt.close()

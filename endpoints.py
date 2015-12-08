from __future__ import division

import numpy as np

from collections import namedtuple
from operator import itemgetter
from itertools import groupby, zip_longest

from scipy.signal import ellipord, ellip, lfilter

#TODO: load links model
#TODO: load echo model

class Outliner(object):

    WINDOW_TYPES = {'Hamming' : np.hamming,
                   'Hanning' : np.hanning,
                   'Blackman' : np.blackman,
                   'Rectangle' : np.ones}
    SMS_TYPE = ('mean', 'median')

    window_prev_links = 70e-3
    harmonic_thresh = 1e-4

    def __init__(self, **kargs):
        self.window_size = kargs.pop('window_size', 3)
        self.frame_rate = kargs.pop('frame_rate', 10000)
        self.chunk_size = kargs.pop('chunk_size', 2)
        self.HPFcutoff = kargs.pop('HPFcutoff', 15)
        self.window_type = kargs.pop('window_type', 'Blackman')
        self.delta_size = kargs.pop('delta_size', 1)
        self.sms = kargs.pop('sms', 'mean')
        self.min_link_len = kargs.pop('min_link_len', 6)
        self.baseline_thresh = kargs.pop('baseline_thresh', 20)
        self.trim_thresh = kargs.pop('trim_thresh', 10)

        if not self.window_type in self.WINDOW_TYPES:
            raise TypeError('Outliner window_type arguments must be one in %s' %
                            self.WINDOW_TYPE)
        self.window = self.WINDOW_TYPES[self.window_type]
        # TODO: type check for self.sms
        # TODO: range check for all other keyword args.
        # TODO: check if there are keywords left in kargs and throw exception

    def extract_features(self, x, fs):
        """
        This function uses rules-based links to outline calls in x
        and also extracts local/global features.

        Inputs:
            x: single-channel audio, int or double
            fs: sampling rate, Hz
        Output:
            Structure of global features, N detected links
        """
        # Find paramters used to calculate spectrogram
        # Samples
        frame_size = np.round(self.window_size / 1000 * fs)
        # Increase FFT size for interpolation
        fft_size = 2**(np.ceil(np.log2(frame_size)) + 2)
        # spectrogram row, rows 1..hpfRow removed
        # from spectrogram for speed/memory
        hpf_row = np.round(self.HPFcutoff * 1e3 / fs * fft_size)

        # Find number of chunks to process, non-overlapping:
        #  Last bit in x used in last chunk
        num_chunks = np.max(1, np.int(x.shape[0]/fs/self.chunk_size))

        # Init spectrogram
        ham_window = self.window(self.frame_size)
        # Process each chunk

    def extract_links(self, x, fs):
        # samples/frame, fractional
        frame_incr = fs / self.frame_rate

        for i in range(0, len(x), fs * self.chunk_size):
            # Get spectrogram/link of chunk
            x1 = x[i:i+step]
            x1 = x1 - numpy.mean(x1)
            # Skip chunk if all zeros
            if np.sum(np.abs(x1)) == 0:
                continue
            # Number of frames in spectrogram
            num_columns = np.ceil(len(x1) / frame_incr)
            # Zero-pad to fit in m*l matrix
            # TODO: translate line [196, end]



if __name__ == "__main__":
    from scipy.io import wavfile
    import sys

    if len(sys.argv) != 2:
        print('Usage: %s wavfile' % sys.argv[0])

    try:
        fs, data = wavfile.read(sys.argv[1])
    except IOError:
        print("Cannot find file: %s" % sys.argv[1])
        exit()

    outliner = Outliner(HPFcutoff = 15)
    outline = outliner.extract_features(data, fs)

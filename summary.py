from __future__ import division

import numpy as np

from collections import namedtuple
from operator import itemgetter
from itertools import groupby

from scipy.signal import ellipord, ellip, lfilter

def find_contiguous_indices(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda (index, item): index - item):
        group = map(itemgetter(1), group)
        if len(group) > 1:
            ranges.append((group[0], group[-1]))
        else:
            ranges.append((group[0], group[0]))
    return ranges

class Summarizer(object):
    # Parameters:
    window_size = 5 # ms, energy frame
    call_window = 3 # frames, +/- local peak window
    pass_window = 150 # frames, minimum inter-pass interval
    detect_thresh_all = [5, 10, 15, 20, 30, 40, 50] # dB, above the noise floor
    Rp = 2 # dB, passband ripple
    Rs = 80 # dB, stopband suppression for HPF/BPF

    SummaryOutput = namedtuple('SummaryOutput', ['num_calls',
                                                 'num_passes',
                                                 'e_norm',
                                                 't',
                                                 'call_index_all'])

    def __init__(self, LPFcutoff = np.inf, HPFcutoff = 15):
        self.LPFcutoff = LPFcutoff
        self.HPFcutoff = HPFcutoff

    def design_filter(self, fs):
        if np.isinf(self.LPFcutoff):
            N, Ws = ellipord(self.HPFcutoff * 1e3 / fs * 2,
                             max(5 * 1e3 / fs * 2,
                                 (self.HPFcutoff - 5) * 1e3 / fs * 2),
                             self.Rp,
                             self.Rs)
            b, a = ellip(N, self.Rp, self.Rs, Ws, 'high')
        else:
            N, Ws = ellipord([self.HPFcutoff * 1e3 / fs * 2, self.LPFcutoff * 1e3 / fs * 2],
                             [max(5*1e3/fs*2,(self.HPFcutoff-5)*1e3/fs*2),
                              min((fs/2-5e3)/fs*2,(self.LPFcutoff+5)*1e3/fs*2)],
                             self.Rp,
                             self.Rs)
            b, a = ellip(N, self.Rp, self.Rs, Ws)
        return b, a

    def find_energy(self, x, fs):
        b, a = self.design_filter(fs)
        num_frames = np.int((x.shape[0] / fs) / (self.window_size * 1e-3))
        e_frame = np.zeros((num_frames, x.shape[1]))
        L = int(round(self.window_size * 1e-3 * fs))
        for p1 in range(x.shape[1]):
            for p in range(num_frames):
                x_frame = lfilter(b,
                                  a,
                                  x[np.arange(L) +
                                    int(round(p * self.window_size * 1e-3 * fs)),
                                    p1])
                e_frame[p, p1] = np.dot(x_frame, x_frame)

        return e_frame

    def estimate_noise_floor(self, e_frame):
        m_frame = np.median(e_frame, 0)
        for i, val in enumerate(m_frame):
            if val == 0:
                m_frame[i] = np.median(e_frame[e_frame[:, i] > 0, i])
                if np.isnan(m_frame[i]):
                    m_frame[i] = 1
        return m_frame

    def normalize_energy_frame(self, e_frame):
        m_frame = self.estimate_noise_floor(e_frame)
        e_norm = np.divide(e_frame, m_frame) # Noise floor at unity
        e_norm = 10 * np.log10(e_norm) # dB
        e_norm[e_norm < -10] = -10 # clip outliers
        return e_norm

    def count_calls(self, x, fs):
        # Assure that signals in 8bits are signed
        if x.dtype is np.dtype('uint8'):
            x = np.int8(np.int16(x) - 128)

        # Assure the signal is a matrix
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        num_ch = x.shape[1]

        e_frame = self.find_energy(x, fs)
        e_norm = self.normalize_energy_frame(e_frame)

        num_calls = np.zeros((len(self.detect_thresh_all), num_ch))
        num_passes = np.zeros((len(self.detect_thresh_all), num_ch))
        call_index_all = []

        for p1 in range(num_ch):
            for p2, detect_thresh in enumerate(self.detect_thresh_all):

                k = np.flatnonzero(e_norm[:, p1] >= detect_thresh)
                if len(k) > 0:
                    call_index = []
                    kranges = find_contiguous_indices(k)

                    for k_start, k_end in kranges:
                        k_end += 1
                        tmp_index = np.argmax(e_norm[k_start:k_end, p1])
                        tmp_index = tmp_index + k_start # index into e_norm

                        begin_i = max(0, tmp_index - self.call_window)
                        end_i = min(e_norm.shape[0], tmp_index + self.call_window)
                        call_end_value = max(e_norm[begin_i:end_i, p1])
                        if e_norm[tmp_index, p1] == call_end_value:
                            call_index.append(tmp_index)

                        # Memorize call indices
                        call_index_all.append(call_index)
                        # Count calls
                        num_calls[p2, p1] = len(call_index)

                        # Count passes
                        call_index_diff = np.diff(call_index)
                        num_passes[p2, p1] = sum(call_index_diff >= self.pass_window) + 1

        t = (np.arange(1, len(e_frame)+1) - 0.5) * self.window_size * 1e-3 # sec
        return self.SummaryOutput(num_calls, num_passes, e_norm, t, call_index_all)


if __name__ == "__main__":
    from scipy.io import wavfile
    import sys

    if len(sys.argv) != 2:
        print('Usage: %s wavfile' % sys.arv[0])

    try:
        fs, data = wavfile.read(sys.argv[1])
    except IOError:
        print("Cannot find file: %s" % sys.argv[1])
        exit()

    summarizer = Summarizer(HPFcutoff = 15)
    summary = summarizer.count_calls(data, fs)

    for ch in range(summary.num_calls.shape[1]):
        print("Channel %i" % (ch+1))
        print '\tNum calls :', summary.num_calls[:, ch].tolist()
        print '\tNum passes:', summary.num_passes[:, ch].tolist()

from __future__ import division

import numpy as np

from collections import namedtuple
from operator import itemgetter
from itertools import groupby, zip_longest

from scipy.signal import ellipord, ellip, lfilter
from scipy.fftpack import fft

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
        self.window_size = kargs.pop('window_size', 0.3)
        self.frame_rate = kargs.pop('frame_rate', 10000)
        self.chunk_size = kargs.pop('chunk_size', 2)
        self.HPFcutoff = kargs.pop('HPFcutoff', 15)
        self.window_type = kargs.pop('window_type', 'Blackman')
        self.delta_size = kargs.pop('delta_size', 1)
        self.sms = kargs.pop('sms', 'mean')
        self.min_link_len = kargs.pop('min_link_len', 6)
        self.baseline_thresh = kargs.pop('baseline_thresh', 20)
        self.trim_thresh = kargs.pop('trim_thresh', 10)
        self.links_thresh = kargs.pop('links_thresh', -25)

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
        # Assure that signals in 8bits are signed
        if x.dtype is np.dtype('uint8'):
            x = np.int8(np.int16(x) - 128)

        # Assure the signal is a matrix
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        num_ch = x.shape[1]

        # Find paramters used to calculate spectrogram
        # Samples
        self.frame_size = np.int(np.round(self.window_size / 1000 * fs))

        # Increase FFT size for interpolation
        self.fft_size = 2**(np.ceil(np.log2(self.frame_size)) + 2)
        # spectrogram row, rows 1..hpfRow removed
        # from spectrogram for speed/memory
        self.hpf_row = np.int(np.round(self.HPFcutoff * 1e3 / fs * self.fft_size))

        # Find number of chunks to process, non-overlapping:
        #  Last bit in x used in last chunk
        num_chunks = max(1, np.int(x.shape[0]/fs/self.chunk_size))

        # Init spectrogram
        self.ham_window = self.window(self.frame_size)
        # Process each chunk
        for ch in range(num_ch):
            links = self.extract_links(x[:, ch], fs)


    def spectral_mean_subtraction(self, s):
        # Truncate at 5th percentile of non-zero values
        s_cutoff = np.percentile(s[s > 0], 5, interpolation='nearest')
        s[s < s_cutoff] = s_cutoff
        s = 10 * np.log10(s)
        return s - np.dot(np.mean(s, axis=1), np.ones((1, s.shape[1])))

    def median_scaling(self, s):
        s = 10 * np.log10(s)
        s_med = np.median(s)
        s[s < s_med] = s_med
        return s - s_med

    def extract_links(self, x, fs):
        # samples/frame, fractional
        frame_incr = fs / self.frame_rate
        step = fs * self.chunk_size
        for i in range(0, len(x), step):
            # Get spectrogram/link of chunk
            x1 = x[i:i+step]
            x1 = x1 - np.mean(x1)
            # Skip chunk if all zeros
            if np.sum(np.abs(x1)) == 0:
                continue
            # Number of frames in spectrogram
            num_columns = np.int(np.ceil(len(x1) / frame_incr))
            # Zero-pad to fit in m*l matrix
            x2 = np.hstack((x1, np.zeros(self.frame_size)))
            x3 = np.zeros((self.frame_size, num_columns))
            for j in range(num_columns):
                idx = j * frame_incr
                x3[:, j] = x2[idx:idx+self.frame_size] * self.ham_window
            sxx = fft(x3, n=self.fft_size, axis=0)

            # Remove low-freq rows, faster computation, less memory
            sxx = sxx[self.hpf_row:int(self.fft_size/2)+1, :num_columns]
            # Removes residual imag part
            sxx = abs(sxx)**2
            s_time_temp = (np.arange(num_columns) * frame_incr
                           + self.frame_size / 2) / fs

            if self.sms == 'sms':
                # Apply spectral mean subtraction
                sxx = self.spectral_mean_subtraction(sxx)
            elif self.sms == 'mean':
                # apply median scaling
                sxx = self.median_scaling(sxx)

            #TODO refctor the name of these variables
            xall_X = sxx
            xall_f = (np.arange(sxx.shape[0]) + self.hpf_row)\
                     * fs / self.fft_size * 1e-3
            xall_t = s_time_temp * 1e3
            # TODO: translate line [227, end]
            # output_links = links(...)
            output_links = find_links(xall_X, xall_f, xall_t)

    def find_links(X, f, t):
        m, n = X.shape
        if len(f) != m or len(t) != n:
            raise Exception("ERROR: size mismatch between X and f and t.")

        local_peaks = numpy.zeros(X.shape)
        local_peaks[1:m-1, :] =  (X[1:m-1, :] >= X[:m-2, :] and
                                  X[1:m-1, :] >  X[2:, :]   and
                                  X[1:m-1, :] >= self.trim_thresh)
        # Init smoothness variables:
        deltaSize = 1;
        ## generic abscissa matrix, [sec,unity]
        z = numpy.ones((2*deltaSize + 1, 2))
        z[:, 0] = numpy.arange(-deltaSize, deltaSize+1) * (t[1] - t[0]) * 1e-3
        C = numpy.dot(numpy.linalg.inv(numpy.dot(z.T, z)), z.T);
        C1 = C[0, :]
        A = numpy.dot(z, C)
        B = numpy.dot((A - numpy.eye(2*deltaSize + 1)).T,
                      (A - numpy.eye(2*deltaSize + 1)))

        # Find neighbor to the right and left of each frame:
        ## row index of nn to the right; == 0 if no nn to the right
        nnRight = numpy.zeros(X.shape)
        ## row index of nn to the left; == 0 if no nn to the left
        nnLeft = numpy.zeros(X.shape)

        currentPeaks = numpy.flatnonzero(localPeaks[:, 0])
        rightPeaks = numpy.flatnonzero(localPeaks[:, 1])
        for p in range(1, n-1):
            leftPeaks = currentPeaks
            currentPeaks = rightPeaks
            rightPeaks = numpy.flatnonzero(localPeaks[:, p+1])

            if len(currentPeaks) > 0:
                # right link only
                if len(leftPeaks) == 0 and len(rightPeaks) > 0:
                    neighborPeaks = rightPeaks
                    for peak in currentPeaks:
                        E = numpy.dot(X[peak, p], numpy.ones((1, len(neighborPeaks))))
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p-1])
                        LL = eval_gmm(numpy.vstack((E, dF)), model)
                        b = numpy.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnRight[peak, p] = neighborPeaks[b]

                # left link only
                elif len(leftPeaks) > 0 and len(rightPeaks) == 0:
                    neighborPeaks = leftPeaks
                    for peak in currentPeaks:
                        E = numpy.dot(X[peak, p], numpy.ones((1, len(neighborPeaks))))
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p-1])
                        LL = eval_gmm(numpy.vstack((E, dF)), model)
                        b = numpy.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = neighborPeaks[b]
                # left and right link
                elif len(leftPeaks) > 0 and len(rightPeaks) > 0:
                    bbb, aaa = numpy.meshgrid(range(len(rightPeaks)),
                                              range(len(leftPeaks)))
                    F1 = numpy.vstack((f[currentPeaks[list(aaa.flat)],
                                       numpy.ones(aaa.shape[0] * aaa.shape[1],
                                       f[currentPeaks[list(bbb.flat)]))) * 1e3
                    for peak in currentPeaks:
                        F1[1, :] = f[peak] * 1e3
                        dF = numpy.dot(C1, F1) * 1e-6 #kHz/ms
                        sF = numpy.maximum((40, 10*numpy.log10(numpy.sum(F1 * numpy.dot(B, F1))) / (2 * deltaSize + 1) + 1)) #dB, averaged
                        gmmFeatures = numpy.vstack((X[peak, p] * numpy.ones(len(dF)), dF, sF))
                        LL = eval_gmm(gmmFeatures, model)
                        b = numpy.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = leftPeaks[b % len(leftPeaks)]
                            # TODO: Double check this is rightPeaks and leftPeaks same size?
                            nnRight[peak, p] = rightPeaks[int(b / len(leftPeaks))]

        # Find reciprocal nearest neighbors, link together:
        nnBoth = numpy.zeros(X.shape)
        # link07.m line 187 and onward

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

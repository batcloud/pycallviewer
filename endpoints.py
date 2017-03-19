from __future__ import division

import numpy as np

from fractions import Fraction
from collections import namedtuple
from operator import itemgetter
from itertools import groupby, zip_longest

from scipy.signal import ellipord, ellip, lfilter
from scipy.fftpack import fft
from scipy.io import loadmat

Link = namedtuple('Link', ['gfeat', 'lfeat'])

HarmonicStats = namedtuple('HarmonicStats',
                            ['ratioMean',
                             'ratioVar',
                             'ratioN',
                             'overlapPercentage',
                             'FratioOverlap',
                             'FratioCurrent',
                             'ratioSE'
                            ])

GlobalFeatures = namedtuple('GlobalFeatures',   
                            ['startTime',       # ms
                             'stopTime',        # ms
                             'duration',        # ms
                             'Fmin',            # Hz
                             'Fmax',            # Hz
                             'FPercentile',     # Hz
                             'E',               # dB
                             'FME',             # Hz
                             'FMETime',         # ms
                             'dFmed',           # kHz / ms
                             'dEmed',           # dB / ms
                             'ddFmed',          # kHz / ms / ms
                             'ddEmed',          # dB / ms / ms
                             'sFmed',           # dB
                             'sEmed'            # dB
                            ])

LocalFeatures = namedtuple('LocalFeatures',
                            ['F',               # Hz
                             'time',            # sec
                             'E',               # dB
                             'dF',              # kHz / ms
                             'dE',              # dB / ms
                             'ddF',             # kHz / ms / ms
                             'ddE',             # dB / ms / ms
                             'sF',              # dB
                             'sE',              # dB
                             'echo_energy'      # dB
                            ])

class GaussianModel(object):
    def __init__(self, name, filename=None, LR=False):
        self.model_name = name
        if filename is not None:
            content = loadmat(filename)
            model = content[self.model_name]
            self.mu = model['mu'][0, 0]
            self.sig = model['sig'][0, 0]
            self.sigInv = model['sigInv'][0, 0]
            self.w = model['w'][0, 0]
            self.prefactor = model['prefactor'][0, 0]
            if LR:
                self.mu = self.mu[:2]
                self.sig = self.sig[:2, :2]
                self.sigInv = np.linalg.inv(self.sig)
                self.prefactor = -1/2 * np.log(np.linalg.det(self.sig)) + np.log(self.w)
            else:
                self.mu = self.mu[[0,1,3]]
                self.sig = self.sig[[[0],[1],[3]], [0,1,3]]
                self.sigInv = np.linalg.inv(self.sig)
                self.prefactor = -1/2 * np.log(np.linalg.det(self.sig)) + np.log(self.w)

    def eval(self, x):
        d, N = x.shape
        M = self.mu.shape[1]
        if M == 1:
            xx = x - self.mu
            y = self.prefactor - np.sum(np.dot(self.sigInv, xx) * xx, axis=0)/2
        else:
            raise Exception("Feature yet supported")
            pass
        return y.flatten()

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
        self.fft_size = 2**(self.frame_size.bit_length() + 2)
        # spectrogram row, rows 1..hpfRow removed
        # from spectrogram for speed/memory
        self.hpf_row = np.int(np.round(self.HPFcutoff * 1e3 / fs * self.fft_size))

        # Init spectrogram
        self.ham_window = self.window(self.frame_size)
        # Process each channel
        links_per_channel = [self.extract_links(x[:, ch], fs) for ch in range(num_ch)]
        # import pdb; pdb.set_trace()
        self.filter_echo(links_per_channel[0])

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

    def compute_spectrogram(self, x, fs):
        # samples/frame, fractional
        frame_incr = fs / self.frame_rate

        x1 = x - np.mean(x)
        # Skip chunk if all zeros
        if np.sum(np.abs(x1)) == 0:
            return

        # Number of frames in spectrogram
        num_columns = np.int(np.ceil(len(x1) / frame_incr))
        # Zero-pad to fit in m*l matrix
        x2 = np.hstack((x1, np.zeros(self.frame_size)))
        x3 = np.zeros((self.frame_size, num_columns))
        for j in range(num_columns):
            idx = int(round(j * frame_incr))
            x3[:, j] = x2[idx:idx+self.frame_size] * self.ham_window
        sxx = fft(x3, n=self.fft_size, axis=0)

        # Remove low-freq rows, faster computation, less memory
        sxx = sxx[self.hpf_row:int(self.fft_size/2)+1, :num_columns]
        # Removes residual imag part
        sxx = abs(sxx)**2
        s_time_temp = (np.arange(num_columns) * frame_incr + self.frame_size / 2) / fs

        if self.sms == 'sms':
            # Apply spectral mean subtraction
            sxx = self.spectral_mean_subtraction(sxx)
        elif self.sms == 'mean':
            # apply median scaling
            sxx = self.median_scaling(sxx)

        f = (np.arange(sxx.shape[0]) + self.hpf_row) * fs / self.fft_size * 1e-3
        t = s_time_temp * 1e3
        return sxx, f, t

    def extract_links(self, x, fs):
        links = []

        # Init linear regression variables:
        # generic abscissa matrix, [sec,unity]
        z = np.ones((2*self.delta_size + 1, 2))
        z[:, 0] = np.arange(-self.delta_size, self.delta_size + 1) / self.frame_rate

        # Hz, prefactor
        C = np.dot(np.linalg.inv(np.dot(z.T, z)), z.T)
        # Hz, linear regression slope from first row of C
        C1 = C[0,:]
        A = np.dot(z, C)

        # unitless, used to find sum-of-squares error
        B = A - np.eye(2*self.delta_size+1)
        B = np.dot(B.T, B)

        # Find number of chunks to process, non-overlapping:
        #  Last bit in x used in last chunk
        step = fs * self.chunk_size
        num_chunks = max(1, np.int(x.shape[0]/fs/self.chunk_size))
        # Process each chunk
        for i, x1 in enumerate(np.split(x, range(step, num_chunks*step, step))):
            # Get spectrogram/link of chunk
            sxx, f, t = self.compute_spectrogram(x1, fs)
            for link in self.find_links(sxx, f, t):
                link = np.array(link)
                # Adjust time/frequency for each link
                link[:, 0] = (link[:, 0] + self.hpf_row)/ self.fft_size * fs # Hz
                link[:, 1] = t[link[:, 1].astype(int)] * 1e-3 + i * self.chunk_size # sec

                # Compute local/global features
                links.append(self.compute_link_features(link, B, C1))
        return links

    def compute_derivative(self, X, B, C1, noisy=False):
        n = X.shape[0]
        dX = np.zeros(n)
        sX = np.zeros(n)

        X1 = np.hstack((X[[0]*self.delta_size], X, X[[-1]*self.delta_size]))
        # TODO: find out to what correspond fft_res before trying this
        if noisy:
            X1 = X1 + (np.random.random(X1.shape) - 0.5) * self.fft_res

        for i in range(n):
            segment = X1[i:(i+2*self.delta_size+1)].T
            dX[i] = np.dot(C1, segment)
            sX[i] = np.dot(np.dot(segment.T, B), segment)

        return dX, sX

    def compute_link_features(self, link, B, C1):
        # Get local features:
        F0 = link[:,0].T # Hz, ROW vector
        A0 = link[:,2].T # dB, ROW vector

        dF0, sF0 = self.compute_derivative(F0, B, C1)
        dA0, sA0 = self.compute_derivative(A0, B, C1)
        ddF0, _  = self.compute_derivative(dF0, B, C1)
        ddA0, _  = self.compute_derivative(dA0, B, C1)

        dF0 = dF0/1e6   # Hz/sec --> kHz/ms
        ddF0 = ddF0/1e9 # Hz/sec/sec --> kHz/ms/ms
        dA0 = dA0/1e3   # dB/sec --> dB/ms
        ddA0 = ddA0/1e6 # dB/sec/sec --> dB/ms/ms
        sF0 = np.maximum(40,10*np.log10(sF0/(2*self.delta_size+1)+1)) # linear regression error, dB (averaged)
        sA0 = 10*np.log10(sA0/(2*self.delta_size+1)) # linear regression error, dB (averaged)

        # Save local features
        # F(Hz), time(sec), E(dB),
        # dF(kHz/ms), dE(dB/ms),
        # ddF(kHz/ms/ms), ddE(dB/ms/ms)
        # sF(dB), sE(dB), echo_energy(dB
        local_features = {}
        local_features['F'] = link[:, 0]            # Hz
        local_features['time'] = link[:, 1]         # sec
        local_features['E'] = link[:, 2]            # dB
        local_features['dF'] = dF0                  # kHz / ms
        local_features['dE'] = dA0                  # dB / ms
        local_features['ddF'] = ddF0                # kHz / ms / ms
        local_features['ddE'] = ddA0                # dB / ms / ms
        local_features['sF'] = sF0                  # dB
        local_features['sE'] = sA0                  # dB
        local_features['echo_energy'] = link[:, 3]  # dB

        # Save global features
        # Start time(ms),End time(ms),Duration(ms),Fmin(Hz),
        # Fmax(Hz),F0 percentiles(Hz),FME(Hz),E(dB),FMETime(ms),
        # median dF0(kHz/ms),median dA0(dB/ms),median ddF0(kHz/ms/ms),
        # median ddA0(dB/ms/ms), median sF0(dB), median sA0(dB)
        global_features = {}
        global_features['startTime'] = link[0, 1] * 1e3
        global_features['stopTime']  = link[-1, 1] * 1e3
        global_features['duration']  = global_features['stopTime'] - global_features['startTime']
        global_features['Fmin'] = np.min(F0)
        global_features['Fmax'] = np.max(F0)
        global_features['FPercentile'] = np.percentile(F0, range(10, 100, 10), interpolation='nearest')

        a0_max_idx = np.argmax(A0)
        global_features['E'] = A0[a0_max_idx]
        global_features['FME'] = F0[a0_max_idx]
        global_features['FMETime'] = (link[a0_max_idx, 1] - link[0, 1]) * 1e3
        global_features['dFmed'] = np.median(dF0)
        global_features['dEmed'] = np.median(dA0)
        global_features['ddFmed'] = np.median(ddF0)
        global_features['ddEmed'] = np.median(ddA0)
        global_features['sFmed'] = np.median(sF0)
        global_features['sEmed'] = np.median(sA0)

        return Link(lfeat=LocalFeatures(**local_features),
                    gfeat=GlobalFeatures(**global_features))

    def find_links(self, X, f, t):
        m, n = X.shape
        if len(f) != m or len(t) != n:
            raise Exception("ERROR: size mismatch between X and f and t.")

        localPeaks = np.zeros(X.shape, dtype=bool)
        localPeaks[1:-1,] = ((X[1:-1,] >= X[:-2,]) & (X[1:-1,] > X[2:,]) &
                             (X[1:-1,] >= self.trim_thresh))

        # Init smoothness variables:
        deltaSize = 1;
        ## generic abscissa matrix, [sec,unity]
        z = np.ones((2*deltaSize + 1, 2))
        z[:, 0] = np.arange(-deltaSize, deltaSize+1) * (t[1] - t[0]) * 1e-3
        C = np.dot(np.linalg.inv(np.dot(z.T, z)), z.T)
        C1 = C[0, :]
        A = np.dot(z, C)
        B = np.dot((A - np.eye(2*deltaSize + 1)).T,
                   (A - np.eye(2*deltaSize + 1)))

        # Find neighbor to the right and left of each frame:
        ## row index of nn to the right; ma.masked if no nn to the right
        nnRight = np.ma.zeros(X.shape, dtype=int); nnRight.mask = True
        ## row index of nn to the left; ma.masked if no nn to the left
        nnLeft = np.ma.zeros(X.shape, dtype=int); nnLeft.mask = True

        currentPeaks = np.flatnonzero(localPeaks[:, 0])
        rightPeaks = np.flatnonzero(localPeaks[:, 1])
        for p in range(1, n-1):
            leftPeaks = currentPeaks
            currentPeaks = rightPeaks
            rightPeaks = np.flatnonzero(localPeaks[:, p+1])

            if currentPeaks.size > 0:
                # right link only
                if leftPeaks.size == 0 and rightPeaks.size > 0:
                    neighborPeaks = rightPeaks
                    for peak in currentPeaks:
                        E = X[peak, p] * np.ones(neighborPeaks.size)
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p+1])
                        LL = links_model_lr.eval(np.vstack((E, dF)))
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnRight[peak, p] = neighborPeaks[b]

                # left link only
                elif leftPeaks.size > 0 and rightPeaks.size == 0:
                    neighborPeaks = leftPeaks
                    for peak in currentPeaks:
                        E = X[peak, p] * np.ones(neighborPeaks.size)
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p-1])
                        LL = links_model_lr.eval(np.vstack((E, dF)))
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = neighborPeaks[b]

                # left and right link
                elif leftPeaks.size > 0 and rightPeaks.size > 0:
                    F1 = np.vstack((f[np.tile(leftPeaks, rightPeaks.size)],
                                    np.ones(leftPeaks.size * rightPeaks.size),
                                    f[np.repeat(rightPeaks, leftPeaks.size)])) * 1e3
                    for peak in currentPeaks:
                        F1[1, :] = f[peak] * 1e3
                        dF = np.dot(C1, F1) * 1e-6 #kHz/ms
                        sF = np.maximum(40, 10*np.log10(np.sum(F1 * np.dot(B, F1), axis=0) / (2 * deltaSize + 1) + 1)) #dB, averaged
                        E  = X[peak, p] * np.ones(dF.size)
                        LL = links_model.eval(np.vstack((E, dF, sF)))
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = leftPeaks[b % leftPeaks.size]
                            nnRight[peak, p] = rightPeaks[b // leftPeaks.size]

        # Find reciprocal nearest neighbors, link together:
        nnBoth = np.ma.zeros(X.shape, dtype=int); nnBoth.mask = True
        for p in range(1, n-1):
            currentPeaks = np.flatnonzero(localPeaks[:, p])
            for peak in currentPeaks:
                if not nnRight[peak, p] is np.ma.masked and nnLeft[nnRight[peak, p], p+1] == peak:
                    nnBoth[peak, p] = nnRight[peak, p]
        # link07.m line 198 and onward
        nnRight = nnBoth

        frame_diff = int(round(self.window_prev_links * self.frame_rate))
        for p in range(n-1):
            # Get indices of links starting in current frame
            for g in np.flatnonzero(np.invert(nnRight.mask[:, p])):
                # [FFT bin, frame, dB, mask dB]
                link = [[g, p, X[g, p], None]]
                while not nnRight[link[-1][0], link[-1][1]] is np.ma.masked:
                    frame = [nnRight[link[-1][0], link[-1][1]],
                             link[-1][1] + 1,
                             X[nnRight[link[-1][0], link[-1][1]], link[-1][1]+1],
                             None]
                    link.append(frame)
                    nnRight[link[-2][0], link[-2][1]] = np.ma.masked
                if len(link) >= self.min_link_len:
                    # Find spectral max in previous window
                    for frame in link:
                        start = max(0, frame[1] - frame_diff)
                        stop  = frame[1] + 1
                        frame[3] = np.max(X[frame[0], start:stop])
                    yield link

    def filter_echo(self, links):
        num_links = len(links)
        cost_terms = np.zeros((5, num_links))
        for i, link in enumerate(links):
            sF = link.lfeat.sF
            # Adjust sF, truncate values at 40 so that sF 
            # appears more Gaussian in distribution:
            sF[sF < 40] = 40

            cost_terms[0, i] = (len(link.lfeat) - 1) / self.frame_rate
            cost_terms[1, i] = np.max(link.lfeat.E)
            cost_terms[2, i] = np.median(sF)
            cost_terms[3, i] = np.median(link.lfeat.echo_energy - sF)
            cost_terms[4, i] = np.median(link.lfeat.dF)
         
        # Compute likelihoods:
        # LLcall = echo_model.eval(cost_terms)
        # LLecho = call_model.eval(cost_terms)
        # == 1: link is a call (fundamental or harmonic);
        # == 0 : not a call (echo or harmonic)
        # is_call = LLcall >= (LLecho + self.baseline_thresh)

        endpoints = np.zeros((num_links, 2))
        for i, link in enumerate(links):
            endpoints[i, 0] = link.lfeat.time[0]
            endpoints[i, 1] = link.lfeat.time[-1]

        harmonic_list = []
        for i, link in enumerate(links):
            h = endpoints[i]
            
            overlap_array  = (endpoints[:, 0] <= h[0]) & (endpoints[:, 1] >= h[0])
            overlap_array |= (endpoints[:, 0] >= h[0]) & (endpoints[:, 1] <= h[1])
            overlap_array |= (endpoints[:, 0] <= h[1]) & (endpoints[:, 1] >= h[1])
            overlap_array[i] = 0
            overlapping_calls = np.flatnonzero(overlap_array)

            link_hstats = {}
            if overlapping_calls.size > 0:
                # Determine harmonic ratio mean and variance 
                # w/ all overlapping links:
                ratioMean = np.zeros(overlapping_calls.size)
                ratioVar = np.zeros(overlapping_calls.size)
                ratioN = np.zeros(overlapping_calls.size)
                overlapPercentage = np.zeros(overlapping_calls.size)
                FratioOverlap = np.zeros(overlapping_calls.size)
                FratioCurrent = np.zeros(overlapping_calls.size)
                ratioSE = np.zeros(overlapping_calls.size)

                Fcurrent = link.lfeat.F
                for j, call in enumerate(overlapping_calls):
                    hOverlap = endpoints[call]

                    # Get frequencies of overlapping link
                    Foverlap = links[call].lfeat.F

                    # Compute ratio of overlapping parts
                    FcurrentStart = max(0, int(round((hOverlap[0]-h[0])*self.frame_rate)))
                    FcurrentEnd = min(Fcurrent.size, Fcurrent.size+int(round((hOverlap[1]-h[1])*self.frame_rate)))
                    FoverlapStart = max(0, int(round((h[0]-hOverlap[0])*self.frame_rate)))
                    FoverlapEnd = min(Foverlap.size, Foverlap.size+int(round((h[1]-hOverlap[1])*self.frame_rate)))
                    Fratio = Foverlap[FoverlapStart:FoverlapEnd]/Fcurrent[FcurrentStart:FcurrentEnd]

                    if Fratio.size > 1:
                        # more robust to outliers due to untrimmed endpoints
                        ratioMean[j] = np.median(Fratio)
                        if ratioMean[j] > 1 :
                            fraction = Fraction.from_float(ratioMean[j])
                            ratioVar[j] = np.var(Fratio)
                        else:
                            fraction = Fraction.from_float(1./ratioMean[j])
                            ratioVar[j] = np.var(1./Fratio)
                        
                        FratioOverlap[j] = fraction.numerator
                        FratioCurrent[j] = fraction.denominator                            
                        ratioN[j] = Fratio.size
                        overlapPercentage[j] = Fratio.size / min(Fcurrent.size, Foverlap.size) * 100
                        ratioSE[j] = ratioVar[j] / Fratio.size
                    else:
                        ratioMean[j] = np.median(Fratio)
                        ratioVar[j] = 10
                        ratioN[j] = Fratio.size
                        overlapPercentage[j] = 0
                        FratioOverlap[j] = -1
                        FratioCurrent[j] = -1
                        ratioSE[j] = 10
                
                link_hstats['ratioMean'] = ratioMean
                link_hstats['ratioVar'] = ratioVar
                link_hstats['ratioN'] = ratioN
                link_hstats['overlapPercentage'] = overlapPercentage
                link_hstats['FratioOverlap'] = FratioOverlap
                link_hstats['FratioCurrent'] = FratioCurrent
                link_hstats['ratioSE'] = ratioSE

            else:
                link_hstats['ratioMean'] = 1
                link_hstats['ratioVar'] = 0
                link_hstats['ratioN'] = 0
                link_hstats['overlapPercentage'] = 0;
                link_hstats['FratioOverlap'] = []
                link_hstats['FratioCurrent'] = []
                link_hstats['ratioSE'] = []

            harmonic_list.append(HarmonicStats(**link_hstats))

        # Assign harmonic numbers and indeces of minimum harmonic
        # getCallEnpoints line 368 - 411
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    import wavfile
    import sys

    if len(sys.argv) != 2:
        print('Usage: %s wavfile' % sys.argv[0])

    try:
        fs, data = wavfile.read(sys.argv[1])
    except IOError:
        print("Cannot find file: %s" % sys.argv[1])
        exit()

    links_model = GaussianModel('linksModel', '/Users/felix/Documents/batcloud/callviewer 2/callviewer/linksModel.mat')
    links_model_lr = GaussianModel('linksModel', '/Users/felix/Documents/batcloud/callviewer 2/callviewer/linksModel.mat', LR=True)

    outliner = Outliner(HPFcutoff = 15)
    outline = outliner.extract_features(data, fs)

from __future__ import division

import numpy as np

from collections import namedtuple
from operator import itemgetter
from itertools import groupby, zip_longest

from scipy.signal import ellipord, ellip, lfilter
from scipy.fftpack import fft
from scipy.io import loadmat

Link = namedtuple('Link', ['gfeat', 'lfeat'])

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
            xx = x - self.mu[:d, np.zeros(N, dtype=np.int)]
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
        import pdb; pdb.set_trace()

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
        links = []
        # Find number of chunks to process, non-overlapping:
        #  Last bit in x used in last chunk
        num_chunks = max(1, np.int(x.shape[0]/fs/self.chunk_size))
        # Process each chunk
        for i in range(num_chunks):
            # Get spectrogram/link of chunk
            if i < num_chunks - 1:
                x1 = x[i*step:(i+1)*step]
            else:
                x1 = x[i*step:]

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
                idx = int(round(j * frame_incr))
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

            output_links = self.find_links(xall_X, xall_f, xall_t)
            # TODO[DEBUG]: verify result from this line to the return.

            # For each link, find spectral max in previous window
            for j, link in enumerate(output_links):
                # t = np.hstack((link, np.zeros((len(link), 1)))) # [FFT bin,frame,dB,mask dB]
                for frame in link:
                    min_idx = max(0, frame[1] - round(self.window_prev_links * self.frame_rate))
                    frame.append(sxx[frame[0], range(int(min_idx), int(frame[1]+1))].max())
                output_links[j] = np.array(link)

            # Adjust time/frequency for each link
            for link in output_links: # for p=1:length(outputLinks),
                link[:, 0] = (link[:, 0] + self.hpf_row)/ self.fft_size * fs # Hz
                link[:, 1] = s_time_temp[link[:, 1].astype(int)] + i * self.chunk_size # sec

            # Get local/global features:
            # getCallEndpoints18.m line 245 and onward
            links.extend(self.get_features(output_links))
        return links

    def get_features(self, output_links):
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

        links_list = []
        for link in output_links:
            # Get local features:
            F0 = link[:,0].T # Hz, ROW vector
            A0 = link[:,2].T # dB, ROW vector

            # Pad endings
            F1 = np.hstack((F0[[0]*self.delta_size], F0, F0[[-1]*self.delta_size]))
            A1 = np.hstack((A0[[0]*self.delta_size], A0, A0[[-1]*self.delta_size]))

            dF0 = np.zeros(F0.shape[0])  # Slope of F0
            dA0 = np.zeros(A0.shape[0])  # Slope of A0
            ddF0 = np.zeros(F0.shape[0]) # Concavity of F0
            ddA0 = np.zeros(A0.shape[0]) # Concavity of A0
            sF0 = np.zeros(F0.shape[0])  # Smoothness of F0
            sA0 = np.zeros(A0.shape[0])  # Smoothness of A0

            # Hz, dithered for non-zero smoothness
            # TODO: find out to what correspond fft_res before trying this
            # FNoisy = F1+(np.random.random(F1.shape)-.5) * self.fft_res;
            FNoisy = F1 # Hz

            for i in range(len(dF0)):
                F = FNoisy[i:(i+2*self.delta_size+1)].T #  COLUMN vector
                dF0[i] = np.dot(C1, F) #slope for linear regression
                sF0[i] = np.dot(np.dot(F.T, B), F) # sum-of-squares error for linear regression
                F = A1[i:(i+2*self.delta_size+1)].T # COLUMN vector
                dA0[i] = np.dot(C1, F)
                sA0[i] = np.dot(np.dot(F.T, B), F)

            sF0 = np.maximum(40,10*np.log10(sF0/(2*self.delta_size+1)+1)) # linear regression error, dB (averaged)
            sA0 = 10*np.log10(sA0/(2*self.delta_size+1)) # linear regression error, dB (averaged)

            F1 = np.hstack([dF0[[0]*self.delta_size], dF0, dF0[[-1] * self.delta_size]]) # pad endings
            A1 = np.hstack([dA0[[0]*self.delta_size], dA0, dA0[[-1] * self.delta_size]]) # pad endings

            for i in range(len(dF0)):
                F = F1[i:(i+2*self.delta_size+1)].T #  COLUMN vector
                ddF0[i] = np.dot(C1, F) # concavity is slope of slope for linear regression
                F = A1[i:(i+2*self.delta_size+1)].T # COLUMN vector
                ddA0[i] = np.dot(C1, F)

            dF0 = dF0/1e6   # Hz/sec --> kHz/ms
            ddF0 = ddF0/1e9 # Hz/sec/sec --> kHz/ms/ms
            dA0 = dA0/1e3   # dB/sec --> dB/ms
            ddA0 = ddA0/1e6 # dB/sec/sec --> dB/ms/ms

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

            links_list.append(Link(lfeat=LocalFeatures(**local_features),
                                   gfeat=GlobalFeatures(**global_features)))
        return links_list

    def find_links(self, X, f, t):
        m, n = X.shape
        if len(f) != m or len(t) != n:
            raise Exception("ERROR: size mismatch between X and f and t.")

        localPeaks = np.zeros(X.shape)
        localPeaks[1:m-1, :] =  np.logical_and(
                                    np.logical_and(X[1:m-1, :] >= X[:m-2, :],
                                                   X[1:m-1, :] >  X[2:, :]),
                                    X[1:m-1, :] >= self.trim_thresh
                                 )
        # Init smoothness variables:
        deltaSize = 1;
        ## generic abscissa matrix, [sec,unity]
        z = np.ones((2*deltaSize + 1, 2))
        z[:, 0] = np.arange(-deltaSize, deltaSize+1) * (t[1] - t[0]) * 1e-3
        C = np.dot(np.linalg.inv(np.dot(z.T, z)), z.T);
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

            if len(currentPeaks) > 0:
                # right link only
                if len(leftPeaks) == 0 and len(rightPeaks) > 0:
                    neighborPeaks = rightPeaks
                    for peak in currentPeaks:
                        E = np.dot(X[peak, p], np.ones((1, len(neighborPeaks))))
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p+1])
                        LL = links_model_lr.eval(np.vstack((E, dF)))
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnRight[peak, p] = neighborPeaks[b]

                # left link only
                elif len(leftPeaks) > 0 and len(rightPeaks) == 0:
                    neighborPeaks = leftPeaks
                    for peak in currentPeaks:
                        E = np.dot(X[peak, p], np.ones((1, len(neighborPeaks))))
                        dF = (f[peak] - f[neighborPeaks]) / (t[p] - t[p-1])
                        LL = links_model_lr.eval(np.vstack((E, dF)))
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = neighborPeaks[b]

                # left and right link
                elif len(leftPeaks) > 0 and len(rightPeaks) > 0:
                    aaa, bbb = np.meshgrid(range(len(leftPeaks)),
                                           range(len(rightPeaks)))
                    F1 = np.vstack((f[leftPeaks[aaa.flatten()]],
                                    np.ones(aaa.shape[0] * aaa.shape[1]),
                                    f[rightPeaks[bbb.flatten()]])) * 1e3
                    for peak in currentPeaks:
                        F1[1, :] = f[peak] * 1e3
                        dF = np.dot(C1, F1) * 1e-6 #kHz/ms
                        sF = np.maximum(40, 10*np.log10(np.sum(F1 * np.dot(B, F1), axis=0) / (2 * deltaSize + 1) + 1)) #dB, averaged
                        gmmFeatures = np.vstack((X[peak, p] * np.ones(len(dF)), dF, sF))
                        LL = links_model.eval(gmmFeatures)
                        b = np.argmax(LL)
                        if LL[b] > self.links_thresh:
                            nnLeft[peak, p] = leftPeaks[b % len(leftPeaks)]
                            nnRight[peak, p] = rightPeaks[int(b / len(leftPeaks))]

        # Find reciprocal nearest neighbors, link together:
        nnBoth = np.ma.zeros(X.shape, dtype=int); nnBoth.mask = True
        for p in range(1, n-1):
            currentPeaks = np.flatnonzero(localPeaks[:, p])
            for peak in currentPeaks:
                if not nnRight[peak, p] is np.ma.masked and nnLeft[nnRight[peak, p], p+1] == peak:
                    nnBoth[peak, p] = nnRight[peak, p]
        # link07.m line 198 and onward
        linkOutput = []
        nnRight = nnBoth

        for p in range(n-1):
            # Get indices of links starting in current frame
            for g in np.flatnonzero(np.invert(nnRight.mask[:, p])):
                tempLink = [[g, p, X[g, p]]]
                while not nnRight[tempLink[-1][0], tempLink[-1][1]] is np.ma.masked:
                    new_row = [nnRight[tempLink[-1][0], tempLink[-1][1]],
                               tempLink[-1][1] + 1,
                               X[nnRight[tempLink[-1][0], tempLink[-1][1]], tempLink[-1][1]+1]]
                    tempLink.append(new_row)
                    nnRight[tempLink[-2][0], tempLink[-2][1]] = np.ma.masked
                if len(tempLink) >= self.min_link_len:
                    linkOutput.append(tempLink)

        return linkOutput

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

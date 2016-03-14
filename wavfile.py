import mmap
import sys
import warnings

import numpy

from scipy.io.wavfile import _read_data_chunk, _read_fmt_chunk, _read_riff_chunk

def read(filename):
    """ Reads common .wav file formats used in bat or bird experiments.

    Mainly based on scipy.io.wavfile functions, to which were added some
    robustness to support Petterson file format.
    """
    with open(filename, 'rb') as fid:
        with mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ) as mfid:
            # Petterson wave file have multiple RIFF headers, find the last one
            rindex = mfid.rfind(b'RIFF')
            if rindex == -1:
                raise ValueError('Missing RIFF tag in wav file.')

            mfid.seek(rindex)
            fsize = _read_riff_chunk(mfid)
            if mfid.read(4) != b'fmt ':
                raise ValueError('Missing format tag in wav file.')

            # Ignore scipy warning on unknown wave file format
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                size, comp, noc, fs, sbytes, ba, bits = _read_fmt_chunk(mfid, is_big_endian=False)

            if mfid.read(4) != b'data':
                raise ValueError('Missing data tag in wav file.')

            x = _read_data_chunk(mfid, comp, noc, bits, is_big_endian=False, mmap=False)
    return fs, x


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit()

    from scipy.io import wavfile

    fs1, x1 = read(sys.argv[1])
    fs2, x2 = wavfile.read(sys.argv[1])

    print(all(x1 == x2))

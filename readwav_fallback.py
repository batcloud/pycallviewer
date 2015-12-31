import sys
import struct

import numpy

def read(filename):
    fid = open(filename, 'rb')
    head = fid.read(60000)

    r = head.rfind(b'RIFF')
    if r == -1:
        raise Exception('Invalid wav file')
    head = head[r:]
    audio_format, num_channels, fs = struct.unpack('<hhi', head[20:28])
    bits_per_sample, = struct.unpack('<h', head[34:36])

    data_ptr = head.rfind(b'data')
    if data_ptr == -1:
        raise Exception('Invalid wav file')

    subchunk2_size, = struct.unpack('<i', head[data_ptr+4:data_ptr+8])

    if audio_format == 1:
        fid.seek(data_ptr + r + 8)
        if bits_per_sample == 8:
            x = numpy.fromstring(fid.read(subchunk2_size), dtype='u1')
        elif bits_per_sample == 16:
            x = numpy.fromstring(fid.read(subchunk2_size), dtype='<i2')

        if num_channels == 2:
            x = x.reshape((-1, 2))
        elif num_channels == 4:
            x = x.reshape((-1, 4))
    else:
        raise Exception('Unsupported wav file')

    return fs, x


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit()

    from scipy.io import wavfile

    fs1, x1 = readwav(sys.argv[1])
    # fs2, x2 = wavfile.read(sys.argv[1])

    import pdb; pdb.set_trace()

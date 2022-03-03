# -*- coding: utf-8 -*-

from scipy.io.wavfile import read as wavread


## helper function: read audio from wav
#
#    @param cAudioFilePath: path to audio file
#
#    @return f_s: sample rate
#    @return x: array with floating point audio data (dimension samples x channels)
def ToolReadAudio(cAudioFilePath):
    [f_s, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        x = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        x = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        x = x - 1.

    return f_s, x

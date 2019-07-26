# -*- coding: utf-8 -*-
"""
helper function: read audio from wav

  Args:
    cAudioFilePath: path to audio file

  Returns:
    sample rate (scalar)
    audio (array)
"""

from scipy.io.wavfile import read as wavread


def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return (samplerate, audio)

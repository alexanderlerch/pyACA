# -*- coding: utf-8 -*-

import numpy as np


## helper function: convert FFT bin to Hz
#
#    @param fBin: FFT bin index (can be float)
#    @param iFftLength: length of the Fft (time domain block size)
#    @param f_s: sample rate of audio data
#
#    @return f: bin frequency (in Hz)
def ToolBin2Freq(fBin, iFftLength, f_s):
    def acaBin2Freq_scalar_I(fBin, iFftLength, f_s):
        return fBin * f_s / float(iFftLength)

    b = np.asarray(fBin)
    if b.ndim == 0:
        return acaBin2Freq_scalar_I(b, iFftLength, f_s)

    f = np.zeros(b.shape)
    for k, bk in enumerate(b):
        f[k] = acaBin2Freq_scalar_I(bk, iFftLength, f_s)
            
    return f

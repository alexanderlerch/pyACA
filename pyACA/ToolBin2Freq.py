# -*- coding: utf-8 -*-
"""
helper function: convert bin to Hz

  Args:
    fBin: FFT bin index (cam be float)
    iFftLength: length of the Fft (time domain block size)
    f_s: sample rate

  Returns:
    bin
"""

import numpy as np


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

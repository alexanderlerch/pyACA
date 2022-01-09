# -*- coding: utf-8 -*-
"""
helper function: convert Hz to bin (still floating point)

  Args:
    fInHz: The frequency to be converted, can be scalar or vector
    iFftLength: length of the Fft (time domain block size)
    f_s: sample rate

  Returns:
    bin
"""

import numpy as np


def ToolFreq2Bin(fInHz, iFftLength, f_s):
    def acaFreq2Bin_scalar_I(f, iFftLength, f_s):
        return f / f_s * iFftLength

    f = np.asarray(fInHz)
    if f.ndim == 0:
        return acaFreq2Bin_scalar_I(f, iFftLength, f_s)

    fBin = np.zeros(f.shape)
    for k, fk in enumerate(f):
        fBin[k] = acaFreq2Bin_scalar_I(fk, iFftLength, f_s)
            
    return fBin

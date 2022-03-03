# -*- coding: utf-8 -*-

import numpy as np


## helper function: normalizes audio signal
#
#    @param x: array with floating point audio data (dimension samples x channels)
#
#    @return x_normx: normalized signal
def ToolNormalizeAudio(x):
    fNorm = np.max(np.abs(x))
    if fNorm == 0:
        fNorm = 1
        
    x_norm = x / fNorm

    return x_norm


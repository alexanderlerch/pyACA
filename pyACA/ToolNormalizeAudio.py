# -*- coding: utf-8 -*-
"""
helper function: normalizes audio signal

  Args:
    x: audio file data (samples x channels)

  Returns:
    x (array): normalized samples
"""

import numpy as np


def ToolNormalizeAudio(x):
    fNorm = np.max(np.abs(x))
    if fNorm == 0:
        fNorm = 1
        
    x_norm = x / fNorm

    return x_norm


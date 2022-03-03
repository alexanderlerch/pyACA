# -*- coding: utf-8 -*-

import numpy as np


## helper function: downmixes an audio signal into one channel
#
#    @param x: array with floating point audio data (dimension samples x channels)
#
#    @return x_downmix: one-channel signal
def ToolDownmix(x):
    
    if x.ndim > 1:
        x_downmix = x.mean(axis=1)
    else:
        x_downmix = x

    return x_downmix

# -*- coding: utf-8 -*-
"""
helper function: convert Hz to Bark scale

  Args:
    fInHz: The frequency to be converted, can be scalar or vector
    cModel: The name of the model ('Schroeder' [default], 'Terhardt', 'Zwicker', 'Traunmuller')

  Returns:
    Bark values of the input dimension 
"""

import numpy as np
import math


def ToolFreq2Bark(fInHz, cModel='Schroeder'):
    def acaSchroeder_scalar(f):
        return 7 * math.asinh(f/650)

    def acaTerhardt_scalar(f):
        return 13.3 * math.atan(0.75 * f/1000)

    def acaZwicker_scalar(f):
        return 13 * math.atan(0.76 * f/1000) + 3.5 * math.atan(f/7500)

    def acaTraunmuller_scalar(f):
        return 26.81/(1+1960./f) - 0.53

    f = np.asarray(fInHz)
    if f.ndim == 0:
        if cModel == 'Terhardt':
            return acaTerhardt_scalar(f)
        elif cModel == 'Zwicker':
            return acaZwicker_scalar(f)
        elif cModel == 'Traunmuller':
            return acaTraunmuller_scalar(f)
        else:
            return acaSchroeder_scalar(f)

    fBark = np.zeros(f.shape)
    if cModel == 'Terhardt':
        for k, fi in enumerate(f):
            fBark[k] = acaTerhardt_scalar(fi)
    elif cModel == 'Zwicker':
        for k, fi in enumerate(f):
            fBark[k] = acaZwicker_scalar(fi)
    elif cModel == 'Traunmuller':
        for k, fi in enumerate(f):
            fBark[k] = acaTraunmuller_scalar(fi)
    else:
        for k, fi in enumerate(f):
            fBark[k] = acaSchroeder_scalar(fi)
            
    return fBark

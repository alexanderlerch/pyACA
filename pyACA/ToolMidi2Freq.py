# -*- coding: utf-8 -*-
"""
helper function: compute periodic von-Hann window

  Args:
    fInHz: The frequency to be converted, can be scalar or vector
    fA4InHz: The reference tuning frequency (default: 440Hz)

  Returns:
    Midi values of the input dimension (floating point)
"""

import numpy as np


def ToolMidi2Freq(pInMidi, fA4InHz = 440):
    def convert_midi2freq_scalar(p, fA4InHz):
 
        if f <= 0:
            return 0
        else:
            return fA4InHz * 2**((p-69) / 12)

    pInMidi = np.asarray(pInMidi)
    if pInMidi.ndim == 0:
       return convert_midi2freq_scalar(pInMidi,fA4InHz)

    fInHz = np.zeros(pInMidi.shape)
    for k,p in enumerate(pInMidi):
        fInHz[k] =  convert_midi2freq_scalar(p,fA4InHz)
            
    return (midi)

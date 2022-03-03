# -*- coding: utf-8 -*-

import numpy as np


## helper function: convert Hz to MIDI pitch (floating point)
#
#    @param fInHz: frequency to be converted, can be scalar or vector
#    @param fA4InHz: The reference tuning frequency (default: 440Hz)
#
#    @return midi: MIDI values (floating point)
def ToolFreq2Midi(fInHz, fA4InHz=440):
    def convert_freq2midi_scalar(f, fA4InHz):
 
        if f <= 0:
            return 0
        else:
            return 69 + 12 * np.log2(f/fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz, fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k, f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f, fA4InHz)
            
    return midi

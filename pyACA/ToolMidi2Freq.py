# -*- coding: utf-8 -*-

import numpy as np


## helper function: convert MIDI to Hz
#
#    @param pInMidi: MIDI pitch
#    @param fA4InHz: The reference tuning frequency (default: 440Hz)
#
#    @return fInHz: frequency in Hz
def ToolMidi2Freq(pInMidi, fA4InHz=440):
    def convert_midi2freq_scalar(p, fA4InHz):
 
        if p < 0:
            return 0
        else:
            return fA4InHz * 2**((p-69) / 12)

    pInMidi = np.asarray(pInMidi)
    if pInMidi.ndim == 0:
        return convert_midi2freq_scalar(pInMidi, fA4InHz)

    fInHz = np.zeros(pInMidi.shape)
    for k, p in enumerate(pInMidi):
        fInHz[k] = convert_midi2freq_scalar(p, fA4InHz)
            
    return fInHz

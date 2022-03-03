# -*- coding: utf-8 -*-
"""
computes the RMS of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vrms rms value (row 1: block-based rms, row 2: single pole approx)
    t time stamp

"""

import numpy as np
import pyACA


## computes the RMS of a time domain signal
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: block length in samples
#    @param iHopLength: hop length in samples
#    @param f_s: sample rate of audio data
#
#    @return vrms:rms value (row 1: block-based rms, row 2: single pole approx)
#    @return t: time stamp
def FeatureTimeRms(x, iBlockLength, iHopLength, f_s):

    T_i = .3 
    alpha = 1 - np.exp(-2.2/f_s/T_i)

    # create blocks
    x_b, t = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, f_s)

    # number of results
    iNumOfBlocks = x_b.shape[0]

    # single pole implementation
    v_sp = filterSP(x**2, alpha)

    # allocate memory
    vrms = np.zeros([2, iNumOfBlocks])

    for n, block in enumerate(x_b):
        i_start = n * iHopLength
        i_stop = np.min([len(x), i_start + iBlockLength])

        # calculate the rms
        vrms[0, n] = np.sqrt(np.dot(block, block) / block.size)
        vrms[1, n] = np.max(np.sqrt(v_sp[i_start:i_stop]))

    # convert to dB
    epsilon = 1e-5  # -100dB

    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)

    return vrms, t


def filterSP(x, alpha):
    
    xf = np.zeros(x.shape)
    xf[0] = alpha * x[0]

    for i in range(1, len(x)):
        xf[i] = alpha * x[i] + (1 - alpha) * xf[i-1]
    
    return xf

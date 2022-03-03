# -*- coding: utf-8 -*-

import numpy as np
import pyACA


## computes the zero crossing rate of a time domain signal
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: block length in samples
#    @param iHopLength: hop length in samples
#    @param f_s: sample rate of audio data
#
#    @return vzc: zero crossing rate
#    @return t: time stamp
def FeatureTimeZeroCrossingRate(x, iBlockLength, iHopLength, f_s):

    # create blocks
    xBlocks, t = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, f_s)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]

    # allocate memory
    vzc = np.zeros(iNumOfBlocks)

    for n, block in enumerate(xBlocks):
        # calculate the zero crossing rate
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(block))))

    return vzc, t

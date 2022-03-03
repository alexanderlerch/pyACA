# -*- coding: utf-8 -*-
"""
computes the standard deviation of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vstd standard deviation
    t time stamp

"""

import numpy as np
import pyACA


## computes the standard deviation of a time domain signal
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: block length in samples
#    @param iHopLength: hop length in samples
#    @param f_s: sample rate of audio data
#
#    @return vstd: standard deviation
#    @return t: time stamp
def FeatureTimeStd(x, iBlockLength, iHopLength, f_s):

    # create blocks
    xBlocks, t = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, f_s)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]

    # allocate memory
    vstd = np.zeros(iNumOfBlocks)

    for n, block in enumerate(xBlocks):
        # calculate the rms
        vstd[n] = np.std(block)

    return vstd, t

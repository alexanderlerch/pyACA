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

def FeatureTimeStd(x, iBlockLength, iHopLength, f_s):

    # create blocks
    xBlocks = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vstd = np.zeros(iNumOfBlocks)

    for n, block in enumerate(xBlocks):
        # calculate the rms
        vstd[n] = np.std(block)

    return vstd, t

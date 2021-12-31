# -*- coding: utf-8 -*-
"""
computes the zero crossing rate of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vzc zero crossing rate
    t time stamp

"""

import numpy as np
import pyACA


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

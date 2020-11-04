# -*- coding: utf-8 -*-
"""
computes the RMS of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vrms autocorrelation maximum
    t time stamp

"""

import numpy as np
import pyACA


def FeatureTimeRms(x, iBlockLength, iHopLength, f_s):

    # create blocks
    xBlocks = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vrms = np.zeros(iNumOfBlocks)

    for n, block in enumerate(xBlocks):
        # calculate the rms
        vrms[n] = np.sqrt(np.dot(block, block) / block.size)

    # convert to dB
    epsilon = 1e-5  # -100dB

    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)

    return vrms, t

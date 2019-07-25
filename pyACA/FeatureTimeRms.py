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
import math


def FeatureTimeRms(x, iBlockLength, iHopLength, f_s):

    # number of results
    iNumOfBlocks = math.ceil(x.size / iHopLength)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vrms = np.zeros(iNumOfBlocks)

    for n in range(0, iNumOfBlocks):

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # calculate the rms
        vrms[n] = np.sqrt(np.dot(x[np.arange(i_start, i_stop + 1)], x[np.arange(i_start, i_stop + 1)]) / (i_stop + 1 - i_start))

    # convert to dB
    epsilon = 1e-5  # -100dB

    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)

    return (vrms, t)

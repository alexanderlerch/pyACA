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
import math


def FeatureTimeZeroCrossingRate(x, iBlockLength, iHopLength, f_s):

    # number of results
    iNumOfBlocks = math.floor((x.size - iBlockLength) / iHopLength + 1)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vzc = np.zeros(iNumOfBlocks)

    for n in range(0, iNumOfBlocks):

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # calculate the zero crossing rate
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(x[np.arange(i_start, i_stop + 1)]))))

    return (vzc, t)

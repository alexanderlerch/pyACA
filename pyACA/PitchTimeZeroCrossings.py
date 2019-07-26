# -*- coding: utf-8 -*-
"""
computes f_0 through zero crossing distances

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
      f frequency
      t time stamp for the frequency value
"""

import numpy as np
import math


def PitchTimeZeroCrossings(x, iBlockLength, iHopLength, f_s):

    # initialize
    iNumOfBlocks = math.ceil(x.size / iHopLength)
    f = np.zeros(iNumOfBlocks)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    for n in range(0, iNumOfBlocks):

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # get current block
        if not x[np.arange(i_start, i_stop + 1)].sum():
            continue
        else:
            x_tmp = x[np.arange(i_start, i_stop + 1)]

        # compute zero crossing indices
        x_tmp = x_tmp[np.arange(0, iBlockLength - 1)] * x_tmp[np.arange(1, iBlockLength)]
        i_tmp = np.diff(np.argwhere(x_tmp < 0), axis=0)

        # average distance of zero crossings indicates half period
        if i_tmp.size:
            f[n] = f_s / np.mean(2 * i_tmp)

    return (f, t)

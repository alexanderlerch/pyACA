# -*- coding: utf-8 -*-
"""
computes f_0 through zero crossing distances

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
      f_0 fundamental frequency estimate
      t time stamp for the frequency value
"""

import numpy as np
import math

from pyACA.ToolBlockAudio import ToolBlockAudio


def PitchTimeZeroCrossings(x, iBlockLength, iHopLength, f_s):

    # initialize
    f_0 = np.zeros(iNumOfBlocks)

    # block audio data
    x_b, t = ToolBlockAudio(x, iBlockLength, iHopLength, f_s)
    iNumOfBlocks = x_b.shape[0]

    for n, block in enumerate(x_b):

        # get current block
        if not block.sum():
             continue
        else:
            x_tmp = block

        # compute zero crossing indices
        x_tmp = x_tmp[np.arange(0, iBlockLength - 1)] * x_tmp[np.arange(1, iBlockLength)]
        i_tmp = np.diff(np.argwhere(x_tmp < 0), axis=0)

        # average distance of zero crossings indicates half period
        if i_tmp.size:
            f_0[n] = f_s / np.mean(2 * i_tmp)

    return f_0, t

# -*- coding: utf-8 -*-
"""
computes the lag of the amdf function

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
      f_0 frequency
      t time stamp for the frequency value
"""

import numpy as np
import math

from pyACA.ToolBlockAudio import ToolBlockAudio


def PitchTimeAmdf(x, iBlockLength, iHopLength, f_s):
    # initialize
    f_max = 2000
    f_min = 50

    # block audio data
    x_b, t = ToolBlockAudio(x, iBlockLength, iHopLength, f_s)
    iNumOfBlocks = x_b.shape[0]

    # allocate memory
    f_0 = np.zeros(iNumOfBlocks)

    eta_min = np.floor(f_s / f_max).astype(int)
    eta_max = np.floor(f_s / f_min).astype(int)

    for n, block in enumerate(x_b):

        # calculate the amdf if non zero
        if not block.sum():
            continue
        else:
            afCorr = computeAmdf(block, eta_max)

        # find the coefficients specified in eta
        f_0[n] = np.argmin(afCorr[np.arange(eta_min + 1, afCorr.size)]) + 1

        # convert to Hz
        f_0[n] = f_s / (f_0[n] + eta_min + 1)

    return f_0, t


def computeAmdf(x, eta_max):
    K = x.shape[0]

    if K <= 0:
        return 0

    afAmdf = np.ones(K)

    for eta in range(0, np.min([K, eta_max + 1])):
        afAmdf[eta] = np.sum(np.abs(x[np.arange(0, K - 1 - eta)] - x[np.arange(eta + 1, K)])) / K

    return afAmdf

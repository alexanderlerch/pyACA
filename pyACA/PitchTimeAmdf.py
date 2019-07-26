# -*- coding: utf-8 -*-
"""
computes the lag of the amdf function

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


def PitchTimeAmdf(x, iBlockLength, iHopLength, f_s):
    # initialize
    f_max = 2000
    f_min = 50
    iNumOfBlocks = math.ceil(x.size / iHopLength)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    f = np.zeros(iNumOfBlocks)

    eta_min = int(round(f_s / f_max)) - 1
    eta_max = int(round(f_s / f_min)) - 1

    for n in range(0, iNumOfBlocks):

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # calculate the acf
        if not x[np.arange(i_start, i_stop + 1)].sum():
            continue
        else:
            x_tmp = x[np.arange(i_start, i_stop + 1)]
            afCorr = computeAmdf(x_tmp, eta_max)

        # find the coefficients specified in eta
        f[n] = np.argmin(afCorr[np.arange(eta_min + 1, afCorr.size)]) + 1

        # convert to Hz
        f[n] = f_s / (f[n] + eta_min + 1)

    return (f, t)


def computeAmdf(x, eta_max):
    K = x.shape[0]

    if K <= 0:
        return 0

    afAmdf = np.ones(K)

    for eta in range(0, np.min([K, eta_max + 1])):
        afAmdf[eta] = np.sum(np.abs(x[np.arange(0, K - 1 - eta)] - x[np.arange(eta + 1, K)])) / K

    return (afAmdf)

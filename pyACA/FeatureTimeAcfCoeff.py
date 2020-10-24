# -*- coding: utf-8 -*-
"""
computes the ACF coefficients of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)
    eta: index (or vector of indices) of coeff result

  Returns:
    vacf autocorrelation coefficient
    t time stamp
"""

import numpy as np
import math


def FeatureTimeAcfCoeff(x, iBlockLength, iHopLength, f_s, eta=19):

    # number of results
    iNumOfBlocks = math.floor((x.size - iBlockLength) / iHopLength + 1)
    if (np.isscalar(eta)):
        iNumOfResultsPerBlock = 1
    else:
        iNumOfResultsPerBlock = eta.size

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vacf = np.zeros([iNumOfResultsPerBlock, iNumOfBlocks])

    for n in range(0, iNumOfBlocks):
        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # calculate the acf
        if not x[np.arange(i_start, i_stop + 1)].sum():
            vacf[np.arange(0, iNumOfResultsPerBlock), n] = np.zeros(iNumOfResultsPerBlock)
            continue
        else:
            x_tmp = x[np.arange(i_start, i_stop + 1)]
            afCorr = np.correlate(x_tmp, x_tmp, "full") / np.dot(x_tmp, x_tmp)

        # find the coefficients specified in eta
        vacf[np.arange(0, iNumOfResultsPerBlock), n] = afCorr[iBlockLength + eta]

    return (vacf, t)

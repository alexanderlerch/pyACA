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
import pyACA


def FeatureTimeAcfCoeff(x, iBlockLength, iHopLength, f_s, eta=19):

    # create blocks
    xBlocks = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]
    if (np.isscalar(eta)):
        iNumOfResultsPerBlock = 1
    else:
        iNumOfResultsPerBlock = eta.size

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vacf = np.zeros([iNumOfResultsPerBlock, iNumOfBlocks])

    for n, block in enumerate(xBlocks):
        # calculate the acf
        if not block.sum():
            vacf[np.arange(0, iNumOfResultsPerBlock), n] = np.zeros(iNumOfResultsPerBlock)
            continue
        else:
            afCorr = np.correlate(block, block, "full") / np.dot(block, block)

        # find the coefficients specified in eta
        vacf[np.arange(0, iNumOfResultsPerBlock), n] = afCorr[iBlockLength + eta]

    return vacf, t

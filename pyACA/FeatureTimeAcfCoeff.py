# -*- coding: utf-8 -*-

import numpy as np
import pyACA


## computes the ACF coefficients of a time domain signal
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: block length in samples
#    @param iHopLength: hop length in samples
#    @param f_s: sample rate of audio data
#    @param eta: index (or vector of indices) of coeff result
#
#    @return vacf: autocorrelation coefficient
#    @return t: time stamp
def FeatureTimeAcfCoeff(x, iBlockLength, iHopLength, f_s, eta=19):

    # create blocks
    xBlocks, t = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, f_s)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]
    if np.isscalar(eta):
        iNumOfResultsPerBlock = 1
    else:
        iNumOfResultsPerBlock = eta.size

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

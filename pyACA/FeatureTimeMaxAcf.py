# -*- coding: utf-8 -*-
"""
computes the ACF maxima of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vta autocorrelation maximum
    t time stamp

"""

import numpy as np
import pyACA


def FeatureTimeMaxAcf(x, iBlockLength, iHopLength, f_s, f_max=2000, fMinThresh=0.35):

    # create blocks
    x_b, t = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, f_s)

    # number of results
    iNumOfBlocks = x_b.shape[0]

    # allocate memory
    vacf = np.zeros(iNumOfBlocks)

    for n, block in enumerate(x_b):
        eta_min = np.floor(f_s / f_max).astype(int)

        # calculate the acf if nonzero
        if not block.sum():
            continue
        else:
            afCorr = np.correlate(block, block, "full") / np.dot(block, block)

        afCorr = afCorr[np.arange(iBlockLength, afCorr.size)]

        # update eta_min to avoid main lobe
        eta_tmp = np.argmax(afCorr < fMinThresh)
        eta_min = np.max([eta_min, eta_tmp])

        afDeltaCorr = np.diff(afCorr)
        eta_tmp = np.argmax(afDeltaCorr > 0)
        eta_min = np.max([eta_min, eta_tmp])

        # find the coefficients specified in eta
        vacf[n] = np.max(afCorr[np.arange(eta_min + 1, afCorr.size)])

    return vacf, t

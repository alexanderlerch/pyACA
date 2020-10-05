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
import math


def FeatureTimeMaxAcf(x, iBlockLength, iHopLength, f_s, f_max=2000, fMinThresh=0.35):

    # initialize
    iNumOfBlocks = math.ceil(x.size / iHopLength)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # allocate memory
    vacf = np.zeros(iNumOfBlocks)

    for n in range(0, iNumOfBlocks):
        eta_min = math.floor(f_s / f_max)

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # calculate the acf
        if not x[np.arange(i_start, i_stop + 1)].sum():
            continue
        else:
            x_tmp = x[np.arange(i_start, i_stop + 1)]
            afCorr = np.correlate(x_tmp, x_tmp, "full") / np.dot(x_tmp, x_tmp)

        afCorr = afCorr[np.arange(iBlockLength, afCorr.size)]

        # update eta_min to avoid main lobe
        eta_tmp = np.argmax(afCorr < fMinThresh)
        eta_min = np.max([eta_min, eta_tmp])

        afDeltaCorr = np.diff(afCorr)
        eta_tmp = np.argmax(afDeltaCorr > 0)
        eta_min = np.max([eta_min, eta_tmp])

        # find the coefficients specified in eta
        vacf[n] = np.max(afCorr[np.arange(eta_min + 1, afCorr.size)])

    return (vacf, t)

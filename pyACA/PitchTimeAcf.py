# -*- coding: utf-8 -*-
"""
computes the lag of the autocorrelation function

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


def PitchTimeAcf(x, iBlockLength, iHopLength, f_s):

    # initialize
    f_max = 2000
    fMinThresh = .35

    # block audio data
    x_b, t = ToolBlockAudio(x, iBlockLength, iHopLength, f_s)
    iNumOfBlocks = x_b.shape[0]

    # allocate memory
    f_0 = np.zeros(iNumOfBlocks)

    for n, block in enumerate(x_b):
        eta_min = np.floor(f_s / f_max).astype(int)

        # calculate the acf if non zero
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
        f_0[n] = np.argmax(afCorr[np.arange(eta_min + 1, afCorr.size)]) + 1

        # convert to Hz
        f_0[n] = f_s / (f_0[n] + eta_min + 1)

    return f_0, t

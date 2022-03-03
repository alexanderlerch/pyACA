# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.signal import filtfilt
from scipy.signal import find_peaks

from .ToolGammatoneFb import ToolGammatoneFb


## computes f0 via the "auditory" approach
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: internal block length 
#    @param iHopLength: internal hop length 
#    @param f_s: sample rate of audio data
#
#    @return f_0: fundamental frequency (in Hz)
#    @return t: time stamp
def PitchTimeAuditory(x, iBlockLength, iHopLength, f_s):

    # initialize
    iNumOfBlocks = math.ceil(x.size / iHopLength)
    f_0 = np.zeros(iNumOfBlocks)
    f_max = 2000
    iNumBands = 20
    fLengthLpInS = 0.001

    iLengthLp = math.ceil(fLengthLpInS * f_s)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # apply filterbank
    X = ToolGammatoneFb(x, f_s, iNumBands)

    # half wave rectification
    X[X < 0] = 0

    # smooth the results with a moving average filter
    b = np.ones(iLengthLp) / iLengthLp
    X = filtfilt(b, 1, X)

    for n in range(0, iNumOfBlocks):

        eta_min = int(round(f_s / f_max))
        afSumCorr = np.zeros(iBlockLength - 1)
        x_tmp = np.zeros(iBlockLength)

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # compute ACF per band and summarize
        for k in range(0, iNumBands):
            # get current block
            if X[k, np.arange(i_start, i_stop + 1)].sum() < 1e-20:
                continue
            else:
                x_tmp[np.arange(0, i_stop - i_start + 1)] = X[k, np.arange(i_start, i_stop + 1)]

            afCorr = np.correlate(x_tmp, x_tmp, "full") / np.dot(x_tmp, x_tmp)

            # aggregate bands with simple sum before peak picking
            afSumCorr += afCorr[np.arange(iBlockLength, afCorr.size)]

        if afSumCorr.sum() < 1e-20:
            continue

        # find the highest local maximum
        iPeaks = find_peaks(afSumCorr, height=0)
        if iPeaks[0].size:
            eta_min = np.max([eta_min, iPeaks[0][0] - 1])
        f_0[n] = np.argmax(afSumCorr[np.arange(eta_min + 1, afSumCorr.size)]) + 1

        # convert to Hz
        f_0[n] = f_s / (f_0[n] + eta_min + 1)

    return f_0, t

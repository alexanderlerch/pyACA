# -*- coding: utf-8 -*-

import numpy as np


## helper function: computes transfer functions of MFCC filter bands 
# see function from Slaneys Auditory Toolbox (Matlab)
#
#    @param iFftLength: length of FFT
#    @param f_s: sample rate of audio data
#
#    @return H: matrix with transfer functions
def ToolMfccFb(iFftLength, f_s):

    # initialization
    f_start = 133.3333

    iNumLinFilters = 13
    iNumLogFilters = 27
    iNumFilters = iNumLinFilters + iNumLogFilters

    linearSpacing = 66.66666666
    logSpacing = 1.0711703

    # compute band frequencies
    f = np.zeros(iNumFilters + 2)
    f[np.arange(0, iNumLinFilters)] = f_start + np.arange(0, iNumLinFilters) * linearSpacing
    f[np.arange(iNumLinFilters, iNumFilters + 2)] = f[iNumLinFilters - 1] * logSpacing**np.arange(1, iNumLogFilters + 3)

    # sanity check
    if f[iNumLinFilters - 1] >= f_s / 2:
        f = f[f < f_s / 2]
        iNumFilters = f.shape[0] - 2

    f_l = f[np.arange(0, iNumFilters)]
    f_c = f[np.arange(1, iNumFilters + 1)]
    f_u = f[np.arange(2, iNumFilters + 2)]

    # allocate memory for filters and set max amplitude
    H = np.zeros([iNumFilters, iFftLength])
    afFilterMax = 2 / (f_u - f_l)
    f_k = np.arange(0, iFftLength) / (iFftLength - 1) * f_s / 2

    # compute the transfer functions
    for c in range(0, iNumFilters):
        # lower filter slope
        i_l = np.argmax(f_k > f_l[c])
        i_u = np.max([0, np.argmin(f_k <= f_c[c]) - 1])
        H[c, np.arange(i_l, i_u + 1)] = afFilterMax[c] * (f_k[np.arange(i_l, i_u + 1)] - f_l[c]) / (f_c[c] - f_l[c])
        # upper filter slope
        i_l = i_u + 1
        i_u = np.max([0, np.argmin(f_k < f_u[c]) - 1])
        H[c, np.arange(i_l, i_u + 1)] = afFilterMax[c] * (f_u[c] - f_k[np.arange(i_l, i_u + 1)]) / (f_u[c] - f_c[c])

    return H

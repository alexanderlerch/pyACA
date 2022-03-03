# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import lfilter


## helper function: splits the audio signal into multiple bands via a Gammatone filterbank
# see function from Slaneys Auditory Toolbox (Matlab)
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param f_s: sample rate of audio data
#    @param iNumBands: number of filter bands (default: 20)
#    @param f_low: minimum frequency (default: 100Hz)
#
#    @return X: multi-band signal
def ToolGammatoneFb(x, f_s, iNumBands=20, f_low=100):

    # initialization
    fEarQ = 9.26449
    fBW = 24.7
    iOrder = 1
    T = 1 / f_s

    # allocate output memory
    X = np.zeros([iNumBands, x.shape[0]])

    # compute the mid frequencies
    f_c = getMidFrequencies(f_low, f_s / 2, iNumBands, fEarQ, fBW)

    # compute the coefficients
    [afCoeffB, afCoeffA] = getCoeffs(f_c, 1.019 * 2 * np.pi * (((f_c / fEarQ)**iOrder + fBW**iOrder)**(1 / iOrder)), T)

    # do the (cascaded) filter process
    for k in range(0, iNumBands):
        X[k, :] = x
        for j in range(0, 4):
            X[k, :] = lfilter(afCoeffB[j, :, k], afCoeffA[j, :, k], X[k, :])

    return X


# see function ERBSpace.m from Slaneys Auditory Toolbox
def getMidFrequencies(f_low, f_hi, iNumBands, fEarQ, fBW):

    freq = np.log((f_low + fEarQ * fBW) / (f_hi + fEarQ * fBW)) / iNumBands
    f_c = np.exp(np.arange(1, iNumBands + 1) * freq)
    f_c = -(fEarQ * fBW) + f_c * (f_hi + fEarQ * fBW)

    return f_c


# see function MakeERBFilters.m from Slaneys Auditory Toolbox
def getCoeffs(f_c, B, T):

    fCos = np.cos(2 * f_c * np.pi * T)
    fSin = np.sin(2 * f_c * np.pi * T)
    fExp = np.exp(B * T)
    fSqrtA = 2 * np.sqrt(3 + 2**(3 / 2))
    fSqrtS = 2 * np.sqrt(3 - 2**(3 / 2))

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * fCos / fExp
    B2 = np.exp(-2 * B * T)

    A11 = -(2 * T * fCos / fExp + fSqrtA * T * fSin / fExp) / 2
    A12 = -(2 * T * fCos / fExp - fSqrtA * T * fSin / fExp) / 2
    A13 = -(2 * T * fCos / fExp + fSqrtS * T * fSin / fExp) / 2
    A14 = -(2 * T * fCos / fExp - fSqrtS * T * fSin / fExp) / 2

    fSqrtA = np.sqrt(3 + 2**(3 / 2))
    fSqrtS = np.sqrt(3 - 2**(3 / 2))
    fArg = (f_c * np.pi * T) * 1j

    fExp1 = 2 * np.exp(4 * fArg)
    fExp2 = 2 * np.exp(-(B * T) + 2 * fArg)

    afGain = np.abs((-fExp1 * T + fExp2 * T * (fCos - fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos - fSqrtA * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtA * fSin)) /
                    (-2 / np.exp(2 * B * T) - fExp1 + (2 + fExp1) / fExp)**4)

    # this is Slaney's compact format - now resort into 3D Matrices
    # fcoefs = [A0*ones(length(f_c),1) A11 A12 A13 A14 A2*ones(length(f_c),1) B0*ones(length(f_c),1) B1 B2 afGain];

    afCoeffB = np.zeros([4, 3, B.size])
    afCoeffA = np.zeros([4, 3, B.size])

    for k in range(0, B.size):
        afCoeffB[0, :, k] = [A0, A11[k], A2] / afGain[k]
        afCoeffA[0, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[1, :, k] = [A0, A12[k], A2]
        afCoeffA[1, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[2, :, k] = [A0, A13[k], A2]
        afCoeffA[2, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[3, :, k] = [A0, A14[k], A2]
        afCoeffA[3, :, k] = [B0, B1[k], B2[k]]

    return afCoeffB, afCoeffA

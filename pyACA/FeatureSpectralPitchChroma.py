# -*- coding: utf-8 -*-
"""
computes the pitch chroma from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    v_pc: pitch chroma
"""

import numpy as np
import math


## computes the pitch chroma from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return v_pc: pitch chroma
def FeatureSpectralPitchChroma(X, f_s):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    # generate filter matrix
    H = generatePcFilters(X.shape[0], f_s)

    # compute pitch chroma
    v_pc = np.dot(H, X**2)

    # norm pitch chroma to a sum of 1 but avoid div by zero
    norm = v_pc.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1
    v_pc = v_pc / norm

    return np.squeeze(v_pc) if isSpectrum else v_pc


def generatePcFilters(iSpecLength, f_s):

    # initialization at C4
    f_mid = 261.63
    iNumOctaves = 4
    iNumPitchesPerOctave = 12

    # sanity check
    while f_mid * 2**iNumOctaves > f_s / 2.:
        iNumOctaves = iNumOctaves - 1

    H = np.zeros([iNumPitchesPerOctave, iSpecLength])

    # for each pitch class i create weighting factors in each octave j
    for i in range(0, iNumPitchesPerOctave):
        afBounds = np.array([2**(-1 / (2 * iNumPitchesPerOctave)), 2**(1 / (2 * iNumPitchesPerOctave))]) * f_mid * 2 * (iSpecLength - 1) / f_s
        for j in range(0, iNumOctaves):
            iBounds = np.array([math.ceil(2**j * afBounds[0]), math.ceil(2**j * afBounds[1])])
            H[i, range(iBounds[0], iBounds[1])] = 1 / (iBounds[1] - iBounds[0] + 1)

        # increment to next semi-tone
        f_mid = f_mid * 2**(1 / iNumPitchesPerOctave)

    return H

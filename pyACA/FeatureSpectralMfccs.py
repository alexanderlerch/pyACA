# -*- coding: utf-8 -*-

import numpy as np
from .ToolMfccFb import ToolMfccFb


## computes the MFCCs from the magnitude spectrum (see Slaney)
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#    @param iNumCoeffs: number of coefficients to compute (default: 13)
#
#    @return v_mfcc: mel frequency cepstral coefficients
def FeatureSpectralMfccs(X, f_s, iNumCoeffs=13):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    # allocate memory
    v_mfcc = np.zeros([iNumCoeffs, X.shape[1]])

    # generate filter matrix
    H = ToolMfccFb(X.shape[0], f_s)
    T = generateDctMatrix(H.shape[0], iNumCoeffs)

    for n in range(0, X.shape[1]):
        # compute the mel spectrum
        X_Mel = np.log10(np.dot(H, X[:, n]) + 1e-20)

        # calculate the mfccs
        v_mfcc[:, n] = np.dot(T, X_Mel)

    return np.squeeze(v_mfcc) if isSpectrum else v_mfcc


# see function mfcc.m from Slaneys Auditory Toolbox
def generateDctMatrix(iNumBands, iNumCepstralCoeffs):
    T = np.cos(np.outer(np.arange(0, iNumCepstralCoeffs), (2 * np.arange(0, iNumBands) + 1)) * np.pi / 2 / iNumBands)

    T = T / np.sqrt(iNumBands / 2)
    T[0, :] = T[0, :] / np.sqrt(2)

    return T

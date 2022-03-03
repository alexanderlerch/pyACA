# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import find_peaks


## computes the tonal power ratio from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#    @param G_T: energy threshold
#
#    @return vtpr: tonal power ratio
def FeatureSpectralTonalPowerRatio(X, f_s, G_T=5e-4):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    X = X**2

    fSum = X.sum(axis=0)
    vtpr = np.zeros(fSum.shape)

    for n in range(0, X.shape[1]):
        if fSum[n] < G_T:
            continue

        # find local maxima above the threshold
        afPeaks = find_peaks(X[:, n], height=G_T)

        if not afPeaks[0].size:
            continue

        # calculate ratio
        vtpr[n] = X[afPeaks[0], n].sum() / fSum[n]

    return np.squeeze(vtpr) if isSpectrum else vtpr

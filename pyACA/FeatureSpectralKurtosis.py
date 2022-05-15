# -*- coding: utf-8 -*-

import numpy as np
from .FeatureSpectralCentroid import FeatureSpectralCentroid
from .FeatureSpectralSpread import FeatureSpectralSpread


## computes the spectral kurtosis from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vsk: spectral kurtosis
def FeatureSpectralKurtosis(X, f_s):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    k = np.arange(0, X.shape[0]) 
    # get spectral centroid and spread (mean and std of dist)
    vsc = FeatureSpectralCentroid(X, f_s)  * 2 / f_s * (X.shape[0]-1)
    vss = FeatureSpectralSpread(X, f_s)    * 2 / f_s * (X.shape[0]-1)

    norm = X.sum(axis=0)
    norm[norm == 0] = 1
    vss[vss == 0] = 1

    # compute kurtosis
    vsk = np.zeros(X.shape[1])
    for n in range(0, X.shape[1]):
        vsk[n] = np.dot((k - vsc[n])**4, X[:, n]) / (vss[n]**4 * norm[n])

    return np.squeeze(vsk - 3) if isSpectrum else (vsk - 3)

# -*- coding: utf-8 -*-

import numpy as np

from .FeatureSpectralCentroid import FeatureSpectralCentroid
from .FeatureSpectralSpread import FeatureSpectralSpread


## computes the spectral skewness from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vssk: spectral skewness
def FeatureSpectralSkewness(X, f_s, UseBookDefinition=False):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    if UseBookDefinition:  # not recommended
        # compute mean and standard deviation
        mu_x = np.mean(X, axis=0, keepdims=True)
        std_x = np.std(X, axis=0)

        # remove mean
        X = X - mu_x

        # compute kurtosis
        vssk = np.sum(X**3, axis=0) / (std_x**3 * X.shape[0])
    else:
        f = np.arange(0, X.shape[0]) / (X.shape[0] - 1) * f_s / 2
        # get spectral centroid and spread (mean and std of dist)
        vsc = FeatureSpectralCentroid(X, f_s) 
        vss = FeatureSpectralSpread(X, f_s)   

        norm = X.sum(axis=0)
        norm[norm == 0] = 1
        vss[vss == 0] = 1

        # compute spread
        vssk = np.zeros(X.shape[1])
        for n in range(0, X.shape[1]):
            vssk[n] = np.dot((f - vsc[n])**3, X[:, n]) / (vss[n]**3 * norm[n] )

    return np.squeeze(vssk) if isSpectrum else vssk

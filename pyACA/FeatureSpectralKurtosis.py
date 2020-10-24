# -*- coding: utf-8 -*-
"""
computes the spectral kurtosis from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsk spectral kurtosis
"""

import numpy as np
from .FeatureSpectralCentroid import FeatureSpectralCentroid
from .FeatureSpectralSpread import FeatureSpectralSpread


def FeatureSpectralKurtosis(X, f_s, UseBookDefinition=False):

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
        vsk = np.sum(X**4, axis=0) / (std_x**4 * X.shape[0])
    else:
        f = np.arange(0, X.shape[0]) / (X.shape[0] - 1) * f_s / 2
        # get spectral centroid and spread (mean and std of dist)
        vsc = FeatureSpectralCentroid(X, f_s)  # *2/f_s * (X.shape[0]-1)
        vss = FeatureSpectralSpread(X, f_s)    # *2/f_s * (X.shape[0]-1)

        norm = X.sum(axis=0)
        norm[norm == 0] = 1
        vss[vss == 0] = 1

        # compute kurtosis
        vsk = np.zeros(X.shape[1])
        for n in range(0, X.shape[1]):
            vsk[n] = np.dot((f - vsc[n])**4, X[:, n]) / (vss[n]**4 * norm[n] * X.shape[0])

    return np.squeeze(vsk - 3) if isSpectrum else (vsk - 3)

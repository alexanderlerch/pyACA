# -*- coding: utf-8 -*-
"""
computes the spectral flux from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsf spectral flux
"""

import numpy as np


def FeatureSpectralFlux(X, f_s):

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]

    afDeltaX = np.diff(X, 1, axis=1)

    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]

    return np.squeeze(vsf) if isSpectrum else vsf

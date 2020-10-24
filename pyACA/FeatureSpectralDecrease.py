# -*- coding: utf-8 -*-
"""
computes the spectral decrease from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsk spectral decrease
"""

import numpy as np


def FeatureSpectralDecrease(X, f_s):

    # compute index vector
    kinv = np.arange(0, X.shape[0])
    kinv[0] = 1
    kinv = 1 / kinv

    norm = X[1:].sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    # compute slope
    vsc = np.dot(kinv, X - X[0]) / norm

    return np.squeeze(vsc, axis=0)

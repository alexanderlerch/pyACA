# -*- coding: utf-8 -*-
"""
computes the spectral centroid from the (squared) magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsc spectral centroid (in Hz)
"""

import numpy as np


def FeatureSpectralCentroid(X, f_s):

    # X = X**2 removed for consistency with book

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    vsc = np.dot(np.arange(0, X.shape[0]), X) / norm

    # convert from index to Hz
    vsc = vsc / (X.shape[0] - 1) * f_s / 2

    return np.squeeze(vsc, axis=0)

# -*- coding: utf-8 -*-
"""
computes the spectral rolloff from the magnitude spectrum
  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data
    kappa: cutoff ratio

  Returns:
    vsr spectral rolloff (in Hz)
"""

import numpy as np


def FeatureSpectralRolloff(X, f_s, kappa=0.85):

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.cumsum(X, axis=0) / norm

    vsr = np.argmax(X >= kappa, axis=0)

    # convert from index to Hz
    vsr = vsr / (X.shape[0] - 1) * f_s / 2

    return vsr

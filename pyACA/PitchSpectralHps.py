# -*- coding: utf-8 -*-
"""
computes the maximum of the Harmonic Product Spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    f HPS maximum location (in Hz)
"""

import numpy as np


def PitchSpectralHps(X, f_s):

    # initialize
    iOrder = 4
    f_min = 300
    f = np.zeros(X.shape[1])

    iLen = int((X.shape[0] - 1) / iOrder)
    afHps = X[np.arange(0, iLen), :]
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] - 1)))

    # compute the HPS
    for j in range(1, iOrder):
        X_d = X[::(j + 1), :]
        afHps *= X_d[np.arange(0, iLen), :]

    f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)

    # find max index and convert to Hz
    f = (f + k_min) / (X.shape[0] - 1) * f_s / 2

    return (f)

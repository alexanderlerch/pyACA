# -*- coding: utf-8 -*-
"""
NoveltyFlux

computes the novelty measure per Spectral Flux

  Args:
      X: spectrogram (dimension FFTLength X Observations)
      afAudioData: array with floating point audio data.
      f_s: sample rate

  Returns:
      d_flux novelty measure

"""

import numpy as np


def NoveltyFlux(X, f_s):
    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    afDeltaX = np.diff(X, 1, axis=1)

    # half-wave rectification
    afDeltaX[afDeltaX < 0] = 0

    # flux
    d_flux = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]

    return np.squeeze(d_flux) if isSpectrum else d_flux

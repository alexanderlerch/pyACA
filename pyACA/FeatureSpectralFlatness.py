# -*- coding: utf-8 -*-
"""
computes the spectral flatness from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vtf spectral flatness
"""

import numpy as np


def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    XLog = np.log(X + 1e-20)

    vtf = np.exp(XLog.mean(axis=0, keepdims=True)) / norm

    vtf[X.min(axis=0, keepdims=True) == 0] = 0
    
    return np.squeeze(vtf, axis=0)

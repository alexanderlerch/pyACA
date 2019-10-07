# -*- coding: utf-8 -*-
"""
computes the spectral crest from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    v spectral crest factor
"""


def FeatureSpectralCrestFactor(X, f_s):

    norm = X.sum(axis=0)
    if X.dim[1] ==1:
        if norm == 0:
            norm = 1
    else:
        norm[norm == 0] = 1

    vtsc = X.max(axis=0) / norm

    return (vtsc)

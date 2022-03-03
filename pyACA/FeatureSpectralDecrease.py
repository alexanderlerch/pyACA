# -*- coding: utf-8 -*-

import numpy as np


## computes the spectral decrease from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vsd: spectral decrease
def FeatureSpectralDecrease(X, f_s):

    # compute index vector
    kinv = np.arange(0, X.shape[0])
    kinv[0] = 1
    kinv = 1 / kinv

    norm = X[1:].sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    # compute slope
    vsd = np.dot(kinv, X - X[0]) / norm

    return np.squeeze(vsd, axis=0)

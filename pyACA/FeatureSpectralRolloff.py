# -*- coding: utf-8 -*-

import numpy as np


## computes the spectral rolloff from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#    @param kappa: cutoff ratio (default: 0.85)
#
#    @return vsr: spectral rolloff (in Hz)
def FeatureSpectralRolloff(X, f_s, kappa=0.85):

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.cumsum(X, axis=0) / norm

    vsr = np.argmax(X >= kappa, axis=0)

    # convert from index to Hz
    vsr = vsr / (X.shape[0] - 1) * f_s / 2

    return vsr

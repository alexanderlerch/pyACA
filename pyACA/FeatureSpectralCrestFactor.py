# -*- coding: utf-8 -*-

import numpy as np


## computes the spectral crest from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vtsc: spectral crest
def FeatureSpectralCrestFactor(X, f_s):

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    vtsc = X.max(axis=0) / norm

    return np.squeeze(vtsc, axis=0)

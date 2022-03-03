# -*- coding: utf-8 -*-

import numpy as np


## computes the spectral flatness from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vtf: spectral flatness
def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    XLog = np.log(X + 1e-20)

    vtf = np.exp(XLog.mean(axis=0, keepdims=True)) / norm

    vtf[X.min(axis=0, keepdims=True) == 0] = 0
    
    return np.squeeze(vtf, axis=0)

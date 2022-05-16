# -*- coding: utf-8 -*-

import numpy as np
from .FeatureSpectralCentroid import FeatureSpectralCentroid


## computes the spectral slope from the magnitude spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return vssl: spectral slope
def FeatureSpectralSlope(X, f_s):

    # compute mean
    vsc = FeatureSpectralCentroid(X, f_s) * 2 / f_s * (X.shape[0] - 1)

    # compute index vector
    kmu = np.arange(0, X.shape[0]) - (X.shape[0]+1) / 2

    # compute slope
    X = X - vsc
    vssl = np.dot(kmu, X) / np.dot(kmu, kmu)

    return vssl

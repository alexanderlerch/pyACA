# -*- coding: utf-8 -*-

import numpy as np


## computes the novelty measure intnroduced by Laroche
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return d_lar: novelty measure
def NoveltyLaroche(X, f_s):

    # difference spectrum (set first diff to zero)
    X = np.c_[np.sqrt(X[:, 0]), np.sqrt(X)]

    afDeltaX = np.diff(X, 1, axis=1)

    # half-wave rectification
    afDeltaX[afDeltaX < 0] = 0

    # flux
    d_lar = np.sum(afDeltaX, axis=0) / X.shape[0]

    return d_lar

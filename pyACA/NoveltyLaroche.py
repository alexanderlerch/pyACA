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
    X = np.c_[X[:, 0], np.sqrt(X)]

    afDiff = np.diff(X, 1, axis=1)

    # flux
    d_lar = np.sum(afDiff, axis=0) / X.shape[0]

    return d_lar

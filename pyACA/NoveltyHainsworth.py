# -*- coding: utf-8 -*-

import numpy as np


## computes the novelty measure introduced by Hainsworth
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return d_hai: novelty measure
def NoveltyHainsworth(X, f_s):

    epsilon = 1e-5

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    X[X <= 0] = epsilon

    afDiff = np.log2(X[:, np.arange(1, X.shape[1])] / X[:, np.arange(0, X.shape[1] - 1)])

    # flux
    d_hai = np.sum(afDiff, axis=0) / X.shape[0]

    return d_hai

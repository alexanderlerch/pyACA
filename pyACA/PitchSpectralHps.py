# -*- coding: utf-8 -*-

import numpy as np


## computes f0 via the maximum of the Harmonic Product Spectrum
#
#    @param X: spectrogram (dimension FFTLength X Observations)
#    @param f_s: sample rate of audio data
#
#    @return f_0: fundamental frequency (in Hz)
def PitchSpectralHps(X, f_s):

    # initialize
    iOrder = 4
    f_min = 300
    f_0 = np.zeros(X.shape[1])

    iLen = int((X.shape[0] - 1) / iOrder)
    afHps = X[np.arange(0, iLen), :]
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] - 1)))

    # compute the HPS
    for j in range(1, iOrder):
        X_d = X[::(j + 1), :]
        afHps *= X_d[np.arange(0, iLen), :]

    f_0 = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)

    # find max index and convert to Hz
    f_0 = (f_0 + k_min) / (X.shape[0] - 1) * f_s / 2

    return f_0

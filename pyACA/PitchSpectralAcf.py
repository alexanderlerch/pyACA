# -*- coding: utf-8 -*-
"""
computes the maximum of the spectral autocorrelation function

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    f acf maximum location (in Hz)
"""

import numpy as np
from scipy.signal import find_peaks


def PitchSpectralAcf(X, f_s):

    # initialize
    f_min = 300
    f = np.zeros(X.shape[1])

    # use spectral symmetry for robustness
    X[0, :] = np.max(X)
    X = np.concatenate((np.flipud(X), X), axis=0)

    # compute the ACF
    for n in range(0, X.shape[1]):

        if X[:, n].sum() < 1e-20:
            continue

        eta_min = int(round(f_min / f_s * (X.shape[0] - 2))) - 1

        afCorr = np.correlate(X[:, n], X[:, n], "full") / np.dot(X[:, n], X[:, n])
        afCorr = afCorr[np.arange(X.shape[0], afCorr.size)]

        # find the highest local maximum
        iPeaks = find_peaks(afCorr, height=0)
        if iPeaks[0].size:
            eta_min = np.max([eta_min, iPeaks[0][0] - 1])
        f[n] = np.argmax(afCorr[np.arange(eta_min, afCorr.size)]) + 1

        # find max index and convert to Hz (note: X has double length)
        f[n] = (f[n] + eta_min) / (X.shape[0] - 2) * f_s

    return (f)

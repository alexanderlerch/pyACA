# -*- coding: utf-8 -*-
"""
NoveltyLaroche
computes the novelty measure used by laroche

  Args:
      X: spectrogram (dimension FFTLength X Observations)
      f_s: sample rate

  Returns:
      d_lar novelty measure

"""

import numpy as np


def NoveltyLaroche(X, f_s):

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], np.sqrt(X)]

    afDiff = np.diff(X, 1, axis=1)

    # flux
    d_lar = np.sum(afDiff, axis=0) / X.shape[0]

    return (d_lar)

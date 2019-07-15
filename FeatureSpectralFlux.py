# -*- coding: utf-8 -*-
"""
computes the spectral flux from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsf spectral flux
"""

import numpy as np
  
    
def FeatureSpectralFlux(X,f_s):   

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:,0],X]
    #X = np.concatenate(X[:,0],X, axis=1)
    afDeltaX = np.diff(X, 1, axis=1)
    
    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis = 0)) / X.shape[0]
  
    return (vsf)
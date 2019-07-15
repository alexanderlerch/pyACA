# -*- coding: utf-8 -*-
"""
computes the spectral decrease from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsk spectral decrease
"""

import numpy as np
  
    
def FeatureSpectralDecrease(X,f_s):   
    
    # compute index vector
    kinv    = np.arange(0,X.shape[0])
    kinv[0] = 1;
    kinv    = 1/kinv;

    norm = X.sum(axis=0,keepdims=True)
    ind = np.argwhere(norm == 0)
    if ind.size:
        norm[norm == 0] = 1 + X[0,ind[0,1]] # hack because I am not sure how to sum subarrays
    norm = norm - X[0,:]
 
    # compute slope
    vsc = np.dot(kinv, X-X[0,:])/norm
    
    return (vsc)
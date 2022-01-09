# -*- coding: utf-8 -*-
"""
computeKey

computes a simple beat histogram
  Args:
      x: array with floating point audio data
      f_s: sample rate

  Returns:
      F: series of subfingerprints
      t: time stamps
"""

import numpy as np

from pyACA.computeFeature import computeFeature
from pyACA.ToolPreprocAudio import ToolPreprocAudio


def computeFingerprint(x, f_s):

    # set default parameters
    target_fs = 5000
    iBlockLength = 2048
    iHopLength = 64
  
    # pre-processing: down-mixing and normalization
    x = ToolPreprocAudio(x)

    # pre-processing: downsampling to target sample rate
    if (f_s != target_fs):
        x = resample(x, target_fs, f_s)
    

    # initialization: generate transformation matrix for 33 frequency bands
    H = generateBands_I(iBlockLength, target_fs)
    
    # initialization: generate FFT window
    afWindow = ToolComputeHann(iBlockLength)
    
    # in the real world, we would do this block by block...
    [X, f, tf] = computeSpectrogram(x, f_s, afWindow, iBlockLength, iHopLength)

    # power spectrum
    X = np.abs(X)**2
    
    # group spectral bins in bands
    E = H * X
    
    # extract fingerprint through diff (both time and freq)
    SubFingerprint = np.diff(np.diff(E,1,1),1,2)
    tf = tf[:-1] + iHopLength / 2 * target_fs

    # quantize fingerprint
    SubFingerprint[SubFingerprint < 0] = 0
    SubFingerprint[SubFingerprint > 0] = 1

    return


def generateBands_I(iFFTLength, f_s)

    # constants
    iNumBands = 33
    f_max = 2000
    f_min = 300
    
    # initialize
    f_band_bounds = f_min * np.exp(np.log(f_max / f_min) * range(iNumBands) / iNumBands)
    f_fft = range(iFFTLength/2+1) / iFFTLength * f_s
    H = np.zeros([iNumBands, iFFTLength/2+1])
    idx = np.zeros([length(f_band_bounds), 2])

    # get indices falling into each band
    for k = 1:length(f_band_bounds)-1
        idx[k, 0] = find(f_fft > f_band_bounds[k], 1, 'first')
        idx[k, 1] = find(f_fft < f_band_bounds[k+1], 1, 'last')
        H[k, idx[k, 0]:idx[k, 1]] = 1
    
    return H
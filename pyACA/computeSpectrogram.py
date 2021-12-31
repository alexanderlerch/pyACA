# -*- coding: utf-8 -*-
"""
computeSpectrogram

computes a spectrogram from the audio data
  Args:
      afAudioData: time domain sample data, dimension channels X samples
      f_s: sample rate of audio data
      bLogarithmic: levels (true) or magnitudes (false)
      afWindow: FFT window of length iBlockLength (default: hann), can be [] empty
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      M: Mel spectrum
      f_c: Center frequencies of mel bands
      t: Timestamps
"""

import numpy as np

from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolBlockAudio import ToolBlockAudio


def computeSpectrogram(afAudioData, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048):

    # Pre-process: down-mix, normalize, zero-pad
    afAudioData = ToolPreprocAudio(afAudioData, iBlockLength)

    if afWindow is None:
        # Compute window function for FFT
        afWindow = ToolComputeHann(iBlockLength)

    assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # block audio data
    xb, t = ToolBlockAudio(afAudioData, iBlockLength, iHopLength, f_s)
    
    # allocate memory
    iSpecDim = np.int_([(xb.shape[1] / 2 + 1), xb.shape[0]])
    X = np.zeros(iSpecDim)

    for n in range(0, xb.shape[0]):
        # apply window
        tmp = abs(np.fft.fft(xb[n, :] * afWindow)) / xb.shape[1]

        # compute magnitude spectrum
        X[:, n] = tmp[range(iSpecDim[0])]

    X[[0, iSpecDim[0]-1], :] = X[[0, iSpecDim[0]-1], :] / np.sqrt(2)  # let's be pedantic about normalization

    f = np.arange(0, iSpecDim[0]) * f_s / iBlockLength

    return (X, f, t)

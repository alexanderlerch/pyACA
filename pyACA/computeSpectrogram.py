# -*- coding: utf-8 -*-
"""
computeSpectrogram

computes a spectrogram from the audio data
  Args:
      x: time domain sample data, dimension channels X samples
      f_s: sample rate of audio data
      afWindow: FFT window of length iBlockLength (default: hann), can be [] empty
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)
      bNormalize: normalize input audio file before fft computation (default: True)
      bMagnitude: return magnitude instead of complex spectrum (default: True)

  Returns:
      X: spectrum
      f: frequencies of bins
      t: timestamps
"""

import numpy as np

from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolBlockAudio import ToolBlockAudio


def computeSpectrogram(x, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048, bNormalize=True, bMagnitude=True):

    iBlockLength = np.int_(iBlockLength)
    iHopLength = np.int_(iHopLength)

    # Pre-process: down-mix, normalize
    x = ToolPreprocAudio(x, bNormalize)

    if afWindow is None:
        # Compute window function for FFT
        afWindow = ToolComputeHann(iBlockLength)

    assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # block audio data
    x_b, t = ToolBlockAudio(x, iBlockLength, iHopLength, f_s)
    
    # allocate memory
    iSpecDim = np.int_([(x_b.shape[1] / 2 + 1), x_b.shape[0]])
    X = np.zeros(iSpecDim)
    if not bMagnitude:
        X = X.astype(complex)

    norm = 2 / x_b.shape[1]

    for n in range(0, x_b.shape[0]):
        # windowed fft
        tmp = np.fft.fft(x_b[n, :] * afWindow) * norm

        # remove redundant spectrum parts
        if bMagnitude:
            X[:, n] = abs(tmp[range(iSpecDim[0])])
        else:
            X[:, n] = tmp[range(iSpecDim[0])]

    # let's be pedantic about normalization
    X[[0, iSpecDim[0]-1], :] = X[[0, iSpecDim[0]-1], :] / np.sqrt(2)

    f = np.arange(0, iSpecDim[0]) * f_s / iBlockLength

    return X, f, t

#######################################################
# main
def computeSpectrogramCl(cPath):
    from pyACA.ToolReadAudio import ToolReadAudio

    # read audio file
    [f_s, x] = ToolReadAudio(cPath)
    
    # for debugging
    iBlockLength = 4096
    iHopLength = 2048

    # compute feature
    [X, f, t] = computeSpectrogram(x, f_s, None, iBlockLength, iHopLength)

    return X, f, t


if __name__ == "__main__":
    import argparse

    # add command line args and parse them
    parser = argparse.ArgumentParser(description='Compute key of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')

    # retrieve command line args
    args = parser.parse_args()
    cPath = args.infile

    # only for debugging
    if __debug__:
        if not cPath:
            cPath = "../ACA-Plots/audio/sax_example.wav"

    # call the function
    computeSpectrogramCl(cPath)

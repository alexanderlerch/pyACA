# -*- coding: utf-8 -*-
"""
computeKey

computes a simple beat histogram
  Args:
      afAudioData: array with floating point audio data.
      f_s: sample rate
      cMethod:  method of beat histogram computation ('Corr' or 'FFT'(default))
      afWindow: FFT window of length iBlockLength (default: hann)
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      beat histogram, BPM axis ticks
"""

import numpy as np

from pyACA.computeSpectrogram import computeSpectrogram
from pyACA.computeNoveltyFunction import computeNoveltyFunction
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolReadAudio import ToolReadAudio


def computeBeatHisto(afAudioData, f_s, cMethod='FFT', afWindow=None, iBlockLength=1024, iHopLength=8):
    # compute window function for FFT
    if afWindow is None:
        afWindow = ToolComputeHann(iBlockLength)

    assert (afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # pre-processing
    afAudioData = ToolPreprocAudio(afAudioData, iBlockLength)

    # novelty function
    [d, t, peaks] = computeNoveltyFunction('Flux', afAudioData, f_s, afWindow, iBlockLength, iHopLength)

    if cMethod == 'Corr':
        # compute autocorrelation of result
        afCorr = np.correlate(d, d, "full") / np.dot(d, d)
        afCorr = afCorr[np.arange(d.shape[0], afCorr.size)]

        Bpm = np.flip(60 / t[np.arange(1, t.shape[0])])
        T = np.flip(afCorr)
    elif cMethod == 'FFT':
        iHistoLength = 65536
        afWindow = ToolComputeHann(iHistoLength)
        f_s = f_s / iHopLength
        if len(d) < 2 * iHistoLength:
            d = [d, np.zeros([1, 2 * iHistoLength - len(d)])]

        [X, f, t] = computeSpectrogram(afAudioData, f_s, None, iBlockLength, iHopLength)

        T = X.mean(axis=1, keepdims=True)

        # restrict range
        T[:8] = 0
        Bpm = f * 60
        lIdx = np.argwhere(Bpm >= 30)[0]
        hIdx = np.argwhere(Bpm > 200)[0]
        T = T[np.arange(lIdx, hIdx)]
        Bpm = Bpm[np.arange(lIdx, hIdx)]
    else:
        T = 0
        Bpm = 0

    return (T, Bpm)


def computeBeatHistoCl(cInPath, cOutPath):
    [f_s, afAudioData] = ToolReadAudio(cInPath)

    [T, Bpm] = computeBeatHisto(afAudioData, f_s)

    result = np.vstack((T, Bpm))

    np.savetxt(cOutPath, result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute simple beat histogram of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')
    parser.add_argument('--outfile', metavar='path', required=False,
                        help='path to output file')

    args = parser.parse_args()
    cInPath = args.infile
    cOutPath = args.outfile

    # only for debugging
    if __debug__:
        if not cInPath:
            cInPath = "../../ACA-Plots/audio/sax_example.wav"
        if not cOutPath:
            cOutPath = "c:/temp/out.txt"

    # call the function
    computeBeatHistoCl(cInPath, cOutPath)

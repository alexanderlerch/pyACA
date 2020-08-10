# -*- coding: utf-8 -*-
"""
computeNoveltyFunction

computes the novelty function for onset detection
 supported novelty measures are:
  'Flux',
  'Laroche',
  'Hainsworth'

  Args:
      cNoveltyName: name of the novelty measure
      afAudioData: array with floating point audio data.
      f_s: sample rate
      afWindow: FFT window of length iBlockLength (default: hann)
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      d novelty function
      t time stamps
      iPeaks indices of picked onset times
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import filtfilt
from scipy.signal import find_peaks

import pyACA
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolReadAudio import ToolReadAudio


def computeNoveltyFunction(cNoveltyName, afAudioData, f_s, afWindow=None, iBlockLength=4096, iHopLength=512):

    # compute window function for FFT
    if afWindow is None:
        afWindow = ToolComputeHann(iBlockLength)

    assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    #mypackage = __import__(".Novelty" + cNoveltyName, package="pyACA")
    hNoveltyFunc = getattr(pyACA, "Novelty" + cNoveltyName)

    # initialization
    fLengthLpInS = 0.3
    iLengthLp = np.max([2, math.ceil(fLengthLpInS * f_s / iHopLength)])

    # pre-processing
    afAudioData = ToolPreprocAudio(afAudioData, iBlockLength)

    # in the real world, we would do this block by block...
    [f, t, X] = spectrogram(afAudioData,
                            f_s,
                            afWindow,
                            iBlockLength,
                            iBlockLength - iHopLength,
                            iBlockLength,
                            False,
                            True,
                            'spectrum')

    #  scale the same as for matlab
    X = np.sqrt(X / 2)

    # novelty function
    d = hNoveltyFunc(X, f_s)

    # smooth novelty function
    b = np.ones(10) / 10
    d = filtfilt(b, 1, d)
    d[d < 0] = 0

    # compute threshold
    b = np.ones(iLengthLp) / iLengthLp
    G_T = .5 * np.mean(d[np.arange(1, d.shape[0])]) + filtfilt(b, 1, d)

    # find local maxima above the threshold
    iPeaks = find_peaks(d - G_T, height=0)

    return (d, t, iPeaks[0])


def computeNoveltyFunctionCl(cPath, cNoveltyName):
    
    [f_s, afAudioData] = ToolReadAudio(cPath)
    # afAudioData = np.sin(2*np.pi * np.arange(f_s*1)*440./f_s)
    [d, t, iPeaks] = computeNoveltyFunction(cNoveltyName, afAudioData, f_s)
    # plot feature output
    if bPlotOutput:
        plt.plot(t, d)
    return (d, t, iPeaks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute key of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')
    parser.add_argument('--noveltyname', metavar='string', required=False,
                        help='novelty measure name in the format NoveltyFlux')
    parser.add_argument('--plotoutput', metavar='bool', required=False,
                        help='option to plot the output')

    # retrieve command line args
    args = parser.parse_args()
    cPath = args.infile
    cNoveltyName = args.noveltyname
    bPlotOutput = args.plotoutput

    # only for debugging
    if __debug__:
        if not cPath:
            cPath = "c:/temp/test.wav"
        if not cNoveltyName:
            cNoveltyName = "Laroche"
        if not bPlotOutput:
            bPlotOutput = True

    # call the function
    computeNoveltyFunctionCl(cPath, cNoveltyName)

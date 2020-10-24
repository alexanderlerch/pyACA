# -*- coding: utf-8 -*-
"""
computePitch

computes the fundamental frequency of the (monophonic) audio
supported pitch trackers are:
    'SpectralAcf',
    'SpectralHps',
    'TimeAcf',
    'TimeAmdf',
    'TimeAuditory',
    'TimeZeroCrossings',
  Args:
      cPitchTrackName: feature to compute, e.g. 'SpectralHps'
      afAudioData: array with floating point audio data.
      f_s: sample rate
      afWindow: FFT window of length iBlockLength (default: hann)
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      f frequency
      t time stamp for the frequency value
"""

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

import pyACA
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolReadAudio import ToolReadAudio


def computePitch(cPitchTrackName, afAudioData, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048):
    
    #mypackage = __import__(".Pitch" + cPitchTrackName, package="pyACA")
    hPitchFunc = getattr(pyACA, "Pitch" + cPitchTrackName)

    # pre-processing
    afAudioData = ToolPreprocAudio(afAudioData, iBlockLength)

    if isSpectral(cPitchTrackName):
        # compute window function for FFT
        if afWindow is None:
            afWindow = ToolComputeHann(iBlockLength)

        assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

        # in the real world, we would do this block by block...
        [f_k, t, X] = spectrogram(afAudioData,
                                  f_s,
                                  afWindow,
                                  iBlockLength,
                                  iBlockLength - iHopLength,
                                  iBlockLength,
                                  False,
                                  True,
                                  'spectrum')

        # we just want the magnitude spectrum...
        X = np.sqrt(X / 2)

        # compute instantaneous pitch chroma
        f = hPitchFunc(X, f_s)

    if isTemporal(cPitchTrackName):
        [f, t] = hPitchFunc(afAudioData, iBlockLength, iHopLength, f_s)

    return (f, t)


#######################################################
# helper functions
def isSpectral(cName):
    bResult = False
    if "Spectral" in cName:
        bResult = True

    return (bResult)


def isTemporal(cName):
    bResult = False
    if "Time" in cName:
        bResult = True

    return (bResult)


def computePitchCl(cPath, cPitchTrackName, bPlotOutput=False):
    
    # read audio file
    [f_s, afAudioData] = ToolReadAudio(cPath)
    # afAudioData = np.sin(2*np.pi * np.arange(f_s*1)*440./f_s)

    # compute feature
    [v, t] = computePitch(cPitchTrackName, afAudioData, f_s)

    # plot feature output
    if bPlotOutput:
        plt.plot(t, v)

    return (v, t)


if __name__ == "__main__":
    import argparse

    # add command line args and parse them
    parser = argparse.ArgumentParser(description='Compute key of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')
    parser.add_argument('--featurename', metavar='string', required=False,
                        help='feature name in the format SpectralPitchChroma')
    parser.add_argument('--plotoutput', metavar='bool', required=False,
                        help='option to plot the output')

    # retrieve command line args
    args = parser.parse_args()
    cPath = args.infile
    cPitchTrackName = args.featurename
    bPlotOutput = args.plotoutput

    # only for debugging
    if __debug__:
        if not cPath:
            cPath = "c:/temp/test.wav"
        if not cPitchTrackName:
            cPitchTrackName = "TimeAmdf"
        if not bPlotOutput:
            bPlotOutput = True

    # call the function
    computePitchCl(cPath, cPitchTrackName, bPlotOutput)

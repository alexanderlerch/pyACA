# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import pyACA
from pyACA.computeSpectrogram import computeSpectrogram
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolReadAudio import ToolReadAudio


## computes a feature from the audio data
# supported features are:
#    'SpectralCentroid',
#    'SpectralCrestFactor',
#    'SpectralDecrease',
#    'SpectralFlatness',
#    'SpectralFlux',
#    'SpectralKurtosis',
#    'SpectralMfccs',
#    'SpectralPitchChroma',
#    'SpectralRolloff',
#    'SpectralSkewness',
#    'SpectralSlope',
#    'SpectralSpread',
#    'SpectralTonalPowerRatio',
#    'TimeAcfCoeff',
#    'TimeMaxAcf',
#    'TimePeakEnvelope',
#    'TimeRms',
#    'TimeStd',
#    'TimeZeroCrossingRate',
#
#    @param cFeatureName: feature to compute, e.g. 'SpectralSkewness'
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param f_s: sample rate of audio data
#    @param afWindow: FFT window of length iBlockLength (default: hann)
#    @param iBlockLength: internal block length (default: 4096 samples)
#    @param iHopLength: internal hop length (default: 2048 samples)
#
#    @return v: feature value
#    @return t: time stamps
def computeFeature(cFeatureName, x, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048):
 
    # mypackage = __import__(".Feature" + cFeatureName, package="pyACA")
    hFeatureFunc = getattr(pyACA, "Feature" + cFeatureName)

    # pre-processing
    x = ToolPreprocAudio(x)

    if isSpectral(cFeatureName):
        # compute window function for FFT
        if afWindow is None:
            afWindow = ToolComputeHann(iBlockLength)

        assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

        # in the real world, we would do this block by block...
        [X, f, t] = computeSpectrogram(x, f_s, None, iBlockLength, iHopLength)

        # compute instantaneous feature
        v = hFeatureFunc(X, f_s)

    if isTemporal(cFeatureName):
        [v, t] = hFeatureFunc(x, iBlockLength, iHopLength, f_s)

    return v, t


#######################################################
# helper functions
def isSpectral(cName):
    bResult = False
    if "Spectral" in cName:
        bResult = True

    return bResult


def isTemporal(cName):
    bResult = False
    if "Time" in cName:
        bResult = True

    return bResult


#######################################################
# main
def computeFeatureCl(cPath, cFeatureName, bPlotOutput=False):

    # read audio file
    [f_s, afAudioData] = ToolReadAudio(cPath)
    
    # for debugging
    iBlockLength = 4096
    iHopLength = 2048

    # compute feature
    [v, t] = computeFeature(cFeatureName, afAudioData, f_s, None, iBlockLength, iHopLength)

    # plot feature output
    if bPlotOutput:
        plt.plot(t, v)

    return v, t


if __name__ == "__main__":
    import argparse

    # add command line args and parse them
    parser = argparse.ArgumentParser(description='Compute feature from wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')
    parser.add_argument('--featurename', metavar='string', required=False,
                        help='feature name in the format SpectralPitchChroma')
    parser.add_argument('--plotoutput', metavar='bool', required=False,
                        help='option to plot the output')

    # retrieve command line args
    args = parser.parse_args()
    cPath = args.infile
    cFeatureName = args.featurename
    bPlotOutput = args.plotoutput

    # only for debugging
    if __debug__:
        if not cPath:
            cPath = "../ACA-Plots/audio/sax_example.wav"
        if not cFeatureName:
            cFeatureName = "SpectralKurtosis"
        if not bPlotOutput:
            bPlotOutput = False

    # call the function
    computeFeatureCl(cPath, cFeatureName, bPlotOutput)

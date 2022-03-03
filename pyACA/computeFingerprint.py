# -*- coding: utf-8 -*-

import numpy as np

from pyACA.computeSpectrogram import computeSpectrogram
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolFreq2Bin import ToolFreq2Bin
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolResample import ToolResample


## computes subfingerprints from audio (derived from Haitsma et al.), 256 subfingerprints comprise one fingerprint
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param f_s: sample rate of audio data
#
#    @return F: series of subfingerprints
#    @return t: time stamps
def computeFingerprint(x, f_s):

    # set default parameters
    fs_target = 5000
    iBlockLength = 2048
    iHopLength = 64
  
    # pre-processing: down-mixing and normalization
    x = ToolPreprocAudio(x)

    # pre-processing: downsampling to target sample rate
    if f_s != fs_target:
        x, t_x = ToolResample(x, fs_target, f_s)
    
    # initialization: generate transformation matrix for 33 frequency bands
    H = generateBands_I(iBlockLength, fs_target)
    
    # initialization: generate FFT window
    afWindow = ToolComputeHann(iBlockLength)
    
    # in the real world, we would do this block by block...
    [X, f, tf] = computeSpectrogram(x, f_s, afWindow, iBlockLength, iHopLength)

    # power spectrum
    X = np.abs(X)**2
    
    # group spectral bins in bands
    E = np.matmul(H, X)
    
    # extract fingerprint through diff (both time and freq)
    SubFingerprint = np.diff(np.diff(E, 1, axis=0), 1, axis=1)
    tf = tf[:-1] + iHopLength / (2 * fs_target)

    # quantize fingerprint
    SubFingerprint[SubFingerprint < 0] = 0
    SubFingerprint[SubFingerprint > 0] = 1

    return SubFingerprint, tf


def generateBands_I(iFftLength, f_s):

    # constants
    iNumBands = 33
    f_max = 2000
    f_min = 300
    
    # initialize
    f_band_bounds = f_min * np.exp(np.log(f_max / f_min) * range(iNumBands+1) / iNumBands)
    f_fft = np.arange(iFftLength / 2 + 1) / iFftLength * f_s
    H = np.zeros([iNumBands, iFftLength // 2 + 1])
    idx = np.zeros([len(f_band_bounds), 2]).astype(int)

    # get indices falling into each band
    for k in range(len(f_band_bounds)-1):
        idx[k, 0] = np.ceil(ToolFreq2Bin(f_band_bounds[k], iFftLength, f_s)).astype(int)
        idx[k, 1] = np.floor(ToolFreq2Bin(f_band_bounds[k+1], iFftLength, f_s)).astype(int)
        H[k, idx[k, 0]:idx[k, 1] + 1] = 1
    
    return H


#######################################################
# main
def computeFingerprintCl(cPath):
    from pyACA.ToolReadAudio import ToolReadAudio

    # read audio file
    [f_s, x] = ToolReadAudio(cPath)
    
    # compute fingerprint
    [F, t] = computeFingerprint(x, f_s)

    return F, t


if __name__ == "__main__":
    import argparse

    # add command line args and parse them
    parser = argparse.ArgumentParser(description='Extract fingerprint from wav file')
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
    computeFingerprintCl(cPath)

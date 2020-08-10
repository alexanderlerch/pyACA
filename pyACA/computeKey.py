# -*- coding: utf-8 -*-
"""
computeKey

computes the musical key of an input audio file
  Args:
      afAudioData: array with floating point audio data.
      f_s: sample rate
      afWindow: FFT window of length iBlockLength (default: hann)
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      key string
"""

import numpy as np
from scipy.signal import spectrogram

from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.FeatureSpectralPitchChroma import FeatureSpectralPitchChroma
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolReadAudio import ToolReadAudio

def computeKey(afAudioData, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048):

    # compute window function for FFT
    if afWindow is None:
        afWindow = ToolComputeHann(iBlockLength)

    assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # key names
    cKeyNames = np.array(['C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj', 'F# Maj', 'G Maj', 'G# Maj', 'A Maj', 'A# Maj', 'B Maj',
                         'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min', 'f# min', 'g min', 'g# min', 'a min', 'a# min', 'b min'])

    # template pitch chroma (Krumhansl major/minor), normalized to a sum of 1
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    t_pc = t_pc / t_pc.sum(axis=1, keepdims=True)

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

    # compute instantaneous pitch chroma
    v_pc = FeatureSpectralPitchChroma(X, f_s)

    # average pitch chroma
    v_pc = v_pc.mean(axis=1)
    # compute manhattan distances for modes (major and minor)
    d = np.zeros(t_pc.shape)
    v_pc = np.concatenate((v_pc, v_pc), axis=0).reshape(2, 12)
    for i in range(0, 12):
        d[:, i] = np.sum(np.abs(v_pc - np.roll(t_pc, i, axis=1)), axis=1)

    # get unwrapped key index
    iKeyIdx = d.argmin()

    cKey = cKeyNames[iKeyIdx]

    return (cKey)


def computeKeyCl(cPath):
    
    [f_s, afAudioData] = ToolReadAudio(cPath)
    # afAudioData = np.sin(2*np.pi * np.arange(f_s*1)*440./f_s)

    cKey = computeKey(afAudioData, f_s)
    print("\ndetected key: ", cKey)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute key of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')

    args = parser.parse_args()
    cPath = args.infile

    # only for debugging
    if not cPath:
        cPath = "c:/temp/test.wav"

    # call the function
    computeKeyCl(cPath)

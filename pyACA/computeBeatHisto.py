# -*- coding: utf-8 -*-

import numpy as np

from pyACA.computeSpectrogram import computeSpectrogram
from pyACA.computeNoveltyFunction import computeNoveltyFunction
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolReadAudio import ToolReadAudio


## computes a simple beat histogram
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param f_s: sample rate of audio data
#    @param cMethod:  method of beat histogram computation ('Corr' or 'FFT'(default))
#    @param afWindow: FFT window of length iBlockLength (Hann will be used if 'None')
#    @param iBlockLength: internal block length (default: 1024 samples)
#    @param iHopLength: internal hop length (default: 8 samples)
#
#    @return T: beat histogram
#    @return Bpm: BPM axis ticks
def computeBeatHisto(x, f_s, cMethod='FFT', afWindow=None, iBlockLength=1024, iHopLength=8):
    # compute window function for FFT
    if afWindow is None:
        afWindow = ToolComputeHann(iBlockLength)

    assert (afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # pre-processing
    x = ToolPreprocAudio(x)

    # novelty function
    [d, t, peaks] = computeNoveltyFunction('Flux', x, f_s, afWindow, iBlockLength, iHopLength)

    if cMethod == 'Corr':
        # compute autocorrelation of result
        r_dd = np.correlate(d, d, "full") / np.dot(d, d)
        r_dd = r_dd[np.arange(d.shape[0], r_dd.size)]

        Bpm = np.flip(60 / t[np.arange(1, t.shape[0])])
        T = np.flip(r_dd)

    elif cMethod == 'FFT':
        # compute the magnitude spectrum of result
        iHistoLength = 65536
        afWindow = np.zeros(2*iHistoLength)
        afWindow[np.arange(0, iHistoLength)] = ToolComputeHann(iHistoLength)
        f_s = f_s / iHopLength
        if len(d) < 2 * iHistoLength:
            d = [d, np.zeros([1, 2 * iHistoLength - len(d)])]

        [D, f, t] = computeSpectrogram(d, f_s, afWindow, 2*iHistoLength, iHistoLength/4)

        T = D.mean(axis=1, keepdims=True)

        # restrict Bpm range
        Bpm = f * 60
        lIdx = np.argwhere(Bpm < 30)[-1]
        hIdx = np.argwhere(Bpm > 200)[0]
        T = T[np.arange(lIdx, hIdx)]
        Bpm = Bpm[np.arange(lIdx, hIdx)]
    else:
        T = 0
        Bpm = 0

    return T, Bpm


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

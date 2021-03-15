# -*- coding: utf-8 -*-
"""
computeMelSpectrogram

computes a mel spectrogram from the audio data
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
from scipy.signal import spectrogram

from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolComputeHann import ToolComputeHann
from pyACA.ToolFreq2Mel import ToolFreq2Mel
from pyACA.ToolMel2Freq import ToolMel2Freq

def computeMelSpectrogram(afAudioData, f_s, afWindow=None, bLogarathmic=True, iBlockLength=4096, iHopLength=2048, iNumMelBands=128, fMax=None):

    if not fMax:
        fMax = f_s/2

    # Pre-process: down-mix, normalize, zero-pad
    afAudioData = ToolPreprocAudio(afAudioData, iBlockLength)

    if afWindow is None:
        # Compute window function for FFT
        afWindow = ToolComputeHann(iBlockLength)

    assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"

    # Compute spectrogram (in the real world, we would do this block by block)
    f, t, X = spectrogram(
        afAudioData,
        fs=f_s,
        window=afWindow,
        nperseg=iBlockLength,
        noverlap=iBlockLength - iHopLength,
        nfft=iBlockLength,
        detrend=False,
        return_onesided=True,
        scaling='spectrum'  # Returns power spectrum
    )

    # Convert power spectrum to magnitude spectrum
    X = np.sqrt(X / 2)

    # Compute Mel filters
    H, f_c = ToolMelFb(iBlockLength, f_s, iNumMelBands, fMax)

    M = np.matmul(H, X)

    if bLogarathmic:
        # Convert amplitude to level (dB)
        M = 20 * np.log10(M + 1e-12)

    return M, f_c, t



def ToolMelFb(iFftLength, f_s, iNumFilters, f_max):

    # Initialization
    f_min = 0
    f_max = min(f_max, f_s/2)
    f_fft = np.linspace(0, f_s/2, iFftLength//2+1)
    H = np.zeros((iNumFilters, f_fft.size))

    # Compute center band frequencies
    mel_min = ToolFreq2Mel(f_min)
    mel_max = ToolFreq2Mel(f_max)
    f_mel = ToolMel2Freq(np.linspace(mel_min, mel_max, iNumFilters+2))

    f_l = f_mel[0:iNumFilters]
    f_c = f_mel[1:iNumFilters + 1]
    f_u = f_mel[2:iNumFilters + 2]

    afFilterMax = 2 / (f_u - f_l)

    # Compute the transfer functions
    for c in range(iNumFilters):
        H[c] = np.logical_and(f_fft > f_l[c], f_fft <= f_c[c]) * \
            afFilterMax[c] * (f_fft-f_l[c]) / (f_c[c]-f_l[c]) + \
            np.logical_and(f_fft > f_c[c], f_fft < f_u[c]) * \
            afFilterMax[c] * (f_u[c]-f_fft) / (f_u[c]-f_c[c])

    return H, f_c

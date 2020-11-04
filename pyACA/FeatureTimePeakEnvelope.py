# -*- coding: utf-8 -*-
"""
computes two peak envelope measures for a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vppm peak envelope (1: max, 2: PPM)
    t time stamp

"""

import numpy as np
import pyACA


def FeatureTimePeakEnvelope(x, iBlockLength, iHopLength, f_s):

    # create blocks
    xBlocks = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength)

    # number of results
    iNumOfBlocks = xBlocks.shape[0]

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    alpha = 1 - np.array([np.exp(-2.2 / (f_s * 0.01)), np.exp(-2.2 / (f_s * 1.5))])

    # allocate memory
    vppm = np.zeros([2, iNumOfBlocks])
    v_tmp = np.zeros(iBlockLength)

    for n, block in enumerate(xBlocks):
        x_block = np.abs(block)

        # detect the maximum per block
        vppm[0, n] = np.max(x_block)

        # calculate the PPM value - take into account block overlaps
        # and discard concerns wrt efficiency
        v_tmp = ppm(x_block, v_tmp[iHopLength - 1], alpha)
        vppm[1, n] = np.max(v_tmp)

    # convert to dB
    epsilon = 1e-5  # -100dB

    vppm[vppm < epsilon] = epsilon
    vppm = 20 * np.log10(vppm)

    return vppm, t


def ppm(x, filterbuf, alpha):

    # initialization
    ppmout = np.zeros(x.shape[0])

    alpha_AT = alpha[0]
    alpha_RT = alpha[1]

    for i in range(0, x.shape[0]):
        if filterbuf > x[i]:
            # release state
            ppmout[i] = (1 - alpha_RT) * filterbuf
        else:
            # attack state
            ppmout[i] = alpha_AT * x[i] + (1 - alpha_AT) * filterbuf

        filterbuf = ppmout[i]

    return (ppmout)

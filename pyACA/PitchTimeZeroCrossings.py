# -*- coding: utf-8 -*-

import numpy as np

from pyACA.ToolBlockAudio import ToolBlockAudio


## computes f0 via zero crossing distances
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: internal block length 
#    @param iHopLength: internal hop length 
#    @param f_s: sample rate of audio data
#
#    @return f_0: fundamental frequency (in Hz)
#    @return t: time stamp
def PitchTimeZeroCrossings(x, iBlockLength, iHopLength, f_s):

    # block audio data
    x_b, t = ToolBlockAudio(x, iBlockLength, iHopLength, f_s)
    iNumOfBlocks = x_b.shape[0]

    # initialize
    f_0 = np.zeros(iNumOfBlocks)

    for n, block in enumerate(x_b):

        # get current block
        if not block.sum():
             continue
        else:
            x_tmp = block

        # compute zero crossing indices
        x_tmp = x_tmp[np.arange(0, iBlockLength - 1)] * x_tmp[np.arange(1, iBlockLength)]
        i_tmp = np.diff(np.argwhere(x_tmp < 0), axis=0)

        # average distance of zero crossings indicates half period
        if i_tmp.size:
            f_0[n] = f_s / np.mean(2 * i_tmp)

    return f_0, t

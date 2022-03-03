# -*- coding: utf-8 -*-

import numpy as np


## helper function: blocks an audio signal into overlapping blocks
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param iBlockLength: internal block length 
#    @param iHopLength: internal hop length 
#    @param f_s: sample rate of audio data
#
#    @return x_b: 2D np.array containing the blocked data of shape (iNumOfBlocks x iBlockLength)
#    @return t: time stamp
def ToolBlockAudio(x, iBlockLength, iHopLength, f_s):

    iNumBlocks = np.ceil(x.shape[0] / iHopLength).astype(int)

    # time stamp vector
    t = np.arange(0, iNumBlocks) * iHopLength / f_s + iBlockLength / (2*f_s)

    # pad with block length zeros just to make sure it runs for weird inputs, too
    afAudioPadded = np.concatenate((x, np.zeros([iBlockLength+iHopLength, ])), axis=0)

    return np.vstack([np.array(afAudioPadded[n*iHopLength:n*iHopLength+iBlockLength]) for n in range(iNumBlocks)]), t

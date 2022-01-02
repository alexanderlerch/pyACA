"""
helper function: Creates blocks from given array. This method creates a block only if there is enough input data to
fill the block. This means if there isn't enough data to fill the last block then that lst chunk of input data will
be discarded. To avoid losing data, you should pad the input with zeros of at least iBlockLength length.

  Args:
    afAudioData: 1D np.array
    iBlockLength: block length
    iHopLength: hop length
    f_s: sample rate

  Returns:
    A 2D np.array containing the blocked data of shape (iNumOfBlocks, iBlockLength).
"""

import numpy as np


def ToolBlockAudio(afAudioData, iBlockLength, iHopLength, f_s):

    iNumBlocks = np.ceil(afAudioData.shape[0] / iHopLength).astype(int)

    # time stamp vector
    t = np.arange(0, iNumBlocks) * iHopLength / f_s

    # pad with block length zeros just to make sure it runs for weird inputs, too
    afAudioPadded = np.concatenate((afAudioData, np.zeros([iBlockLength+iHopLength, ])), axis=0)

    return np.vstack([np.array(afAudioPadded[n*iHopLength:n*iHopLength+iBlockLength]) for n in range(iNumBlocks)]), t

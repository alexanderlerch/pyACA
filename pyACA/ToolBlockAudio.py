"""
helper function: Creates blocks from given array. This method creates a block only if there is enough input data to
fill the block. This means if there isn't enough data to fill the last block then that lst chunk of input data will
be discarded. To avoid losing data, you should pad the input with zeros of at least iBlockLength length.

  Args:
    afAudioData: 1D np.array
    iBlockLength: block length
    iHopLength: hop length

  Returns:
    A 2D np.array containing the blocked data of shape (iNumOfBlocks, iBlockLength).
"""

import numpy as np


def ToolBlockAudio(afAudioData, iBlockLength, iHopLength):

    iNumOfBlocks = np.floor((afAudioData.shape[0] - iBlockLength) / iHopLength + 1).astype(int)

    if iNumOfBlocks < 1:
        return np.array([])
    return np.vstack([np.array(afAudioData[i*iHopLength:i*iHopLength+iBlockLength]) for i in range(iNumOfBlocks)])




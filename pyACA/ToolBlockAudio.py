"""
helper function: Creates blocks from given array.

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




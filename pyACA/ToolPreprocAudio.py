# -*- coding: utf-8 -*-
"""
helper function: read audio from wav

  Args:
    afAudioData: audio file data
    iBlockLength: processing block length

  Returns:
    afAudioData (array): processed samples
"""

import numpy as np


def ToolPreprocAudio(afAudioData, iBlockLength):

    # pre-processing: downmixing
    if afAudioData.ndim > 1:
        afAudioData = afAudioData.mean(axis=1)
    
    # pre-processing: normalization
    fNorm = np.max(np.abs(afAudioData))
    if fNorm != 0:
        afAudioData = afAudioData / fNorm

    # additional preprocessing step might include sample rate conversion and filtering
    
    # pad with block length zeros just to make sure it runs for weird inputs, too
    afAudioData = np.concatenate((afAudioData, np.zeros([iBlockLength, ])), axis=0)
    
    return afAudioData


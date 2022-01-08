# -*- coding: utf-8 -*-
"""
helper function: read audio from wav

  Args:
    afAudioData: audio file data (samples x channels)
    bNormalize: flag to switch off normalization (optional)

  Returns:
    afAudioData (array): processed samples
"""

import numpy as np


def ToolPreprocAudio(afAudioData, bNormalize=True):

    # pre-processing: downmixing
    if afAudioData.ndim > 1:
        afAudioData = afAudioData.mean(axis=1)
    
    # pre-processing: normalization
    if bNormalize:
        fNorm = np.max(np.abs(afAudioData))
        if fNorm != 0:
            afAudioData = afAudioData / fNorm

    # additional preprocessing step might include sample rate conversion and filtering
    
    return afAudioData

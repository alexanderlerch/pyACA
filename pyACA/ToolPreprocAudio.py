# -*- coding: utf-8 -*-
"""
helper function: preprocesses audio signal

  Args:
    x: audio file data (samples x channels)
    bNormalize: flag to switch off normalization (optional)

  Returns:
    afAudioData (array): processed samples
"""

import numpy as np

from pyACA.ToolDownmix import ToolDownmix
from pyACA.ToolNormalizeAudio import ToolNormalizeAudio


def ToolPreprocAudio(x, bNormalize=True):

    # pre-processing: downmixing
    x = ToolDownmix(x)
    
    # pre-processing: normalization
    if bNormalize:
        x = ToolNormalizeAudio(x)
 
    # additional preprocessing step might include sample rate conversion and filtering
    
    return x

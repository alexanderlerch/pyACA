# -*- coding: utf-8 -*-

import numpy as np

from pyACA.ToolDownmix import ToolDownmix
from pyACA.ToolNormalizeAudio import ToolNormalizeAudio


## helper function: pre-processes an audio signal 
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param bNormalize: flag to switch off normalization (default: True)
#
#    @return x_pp: pre-processed signal
def ToolPreprocAudio(x, bNormalize=True):

    # pre-processing: downmixing
    x_pp = ToolDownmix(x)
    
    # pre-processing: normalization
    if bNormalize:
        x_pp = ToolNormalizeAudio(x_pp)
 
    # additional preprocessing step might include sample rate conversion and filtering
    
    return x_pp

# -*- coding: utf-8 -*-
"""
helper function: downmixes audio signal

  Args:
    x: audio file data (samples x channels)

  Returns:
    x_downmix (array): processed samples
"""

import numpy as np


def ToolDownmix(x):
    
    if x.ndim > 1:
        x_downmix = x.mean(axis=1)
    else:
        x_downmix = x

    return x_downmix

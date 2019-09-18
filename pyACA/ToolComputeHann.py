# -*- coding: utf-8 -*-
"""
helper function: compute periodic von-Hann window

  from https://www.programcreek.com/python/example/15589/numpy.hanning

  Args:
    window_length: The number of points in the returned window.

  Returns:
    A 1D np.array containing the periodic hann window.
"""

import numpy as np


def ToolComputeHann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

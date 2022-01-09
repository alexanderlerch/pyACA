# -*- coding: utf-8 -*-
"""
helper function: compute very bad quality resampling

  Args:
    x: input signal (1D)
    fs_out: output sample rate
    fs_in: input sample rate

  Returns:
    x_out: resampled audio signal
    t_out: corresponding time vector
"""

import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.interpolate import interp1d


def ToolResample(x, fs_out, fs_in):

    if fs_out > fs_in:
        omega_cutoff = fs_in / fs_out
    else:
        omega_cutoff = fs_out / fs_in
    
    # compute filter coefficients
    iOrder = 4
    [b, a] = butter(iOrder, 0.9 * omega_cutoff)

    # time axes
    t_in = np.arange(len(x)) / float(fs_in)
    t_out = np.arange(np.round(t_in[-1] * fs_out)) / float(fs_out)
    
    if fs_out > fs_in:
        # upsample: interpolate and filter
        
        # this uses the most horrible interpolation possible
        hInterpol = interp1d(t_in, x, kind='linear')
        x_out = hInterpol(t_out)
        
        # low pass filter
        x_out = filtfilt(b, a, x_out)
    else:
        # downsample: filter and interpolate
        
        # low pass filter
        x_out = filtfilt(b, a, x)
        
        # this uses the most horrible interpolation possible
        hInterpol = interp1d(t_in, x_out, kind='linear')
        x_out = hInterpol(t_out)

    return x_out, t_out

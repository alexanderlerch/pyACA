# -*- coding: utf-8 -*-
"""
helper function: compute instantaneous frequency for two neighboring blocks of complex spectra

  Args:
    X_complex: audio file data
    iHopLength: processing hop size
    f_s: sample rate

  Returns:
    f_I (array): instantaneous frequency
"""

import numpy as np


def ToolInstFreq(X_complex, iHopLength, f_s):

    # get phase
    phi = np.angle(X_complex)
    
    # phase offset
    omega = np.pi * iHopLength / X_complex.shape[1] * np.arange(0, X_complex.shape[1])

    # unwrapped difference
    delta_phi = omega + princarg_I(phi[1, :] - phi[0, :] - omega)

    # instantaneous frequency
    f_I = delta_phi / iHopLength / (2*np.pi) * f_s
    
    return f_I


def princarg_I(phi):

    phase = np.mod(phi + np.pi, -2*np.pi) + np.pi
    return phase

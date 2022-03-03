# -*- coding: utf-8 -*-

import numpy as np


## computes instantaneous frequency utilizing phase of neighboring blocks of complex spectra
#
#    @param X_complex: complex spectrogram (dimension FFTLength X Observations)
#    @param iHopLength: hop length of spectrogram computation
#    @param f_s: sample rate of audio data
#
#    @return f_I: instantaneous frequency (in Hz)
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

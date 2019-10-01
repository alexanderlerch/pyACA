# -*- coding: utf-8 -*-
"""
NoveltyFlux

computes the novelty measure per Spectral Flux

  Args:
      X: spectrogram (dimension FFTLength X Observations)
      afAudioData: array with floating point audio data.
      f_s: sample rate

  Returns:
      d_flux novelty measure

"""

from .FeatureSpectralFlux import FeatureSpectralFlux


def NoveltyFlux(X, f_s):
    d_flux = FeatureSpectralFlux(X, f_s)

    return (d_flux)

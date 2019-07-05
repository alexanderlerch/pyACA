# -*- coding: utf-8 -*-
"""
computeFeature

computes a feature from the audio data
supported features are:
    'SpectralCentroid',
    'SpectralPitchChroma',
  Args:
      cFeatureName: feature to compute, e.g. 'SpectralSkewness'
      afAudioData: array with floating point audio data.
      f_s: sample rate
      afWindow: FFT window of length iBlockLength (default: hann)
      iBlockLength: internal block length (default: 4096 samples)
      iHopLength: internal hop length (default: 2048 samples)

  Returns:
      feature value v
      time stamp t
"""

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt 
       
       
def computeFeature(cFeatureName, afAudioData, f_s, afWindow=None, iBlockLength=4096, iHopLength=2048):
    from ToolComputeHann import ToolComputeHann

    mypackage = __import__('Feature' + cFeatureName)
    hFeatureFunc = getattr(mypackage, 'Feature' + cFeatureName)

    if isSpectral(cFeatureName):
        # compute window function for FFT
        if afWindow is None:
            afWindow = ToolComputeHann(iBlockLength)
    
        assert(afWindow.shape[0] == iBlockLength), "parameter error: invalid window dimension"
    
        # pre-processing: downmixing 
        if afAudioData.ndim > 1:
            afAudioData = afAudioData.mean(axis=1)
        
        # pre-processing: normalization
        fNorm = np.max(np.abs(afAudioData));
        if fNorm != 0:
            afAudioData = afAudioData/fNorm
   
        afAudioData = np.concatenate((afAudioData,np.zeros([iBlockLength,])),axis = 0)         
        # in the real world, we would do this block by block...
        [f,t,X] = spectrogram(  afAudioData,
                                f_s,
                                afWindow,
                                iBlockLength,
                                iBlockLength - iHopLength,
                                iBlockLength,
                                False,
                                True,
                                'spectrum')
    
        #  scale the same as for matlab
        X = np.sqrt(X/2)
    
        # compute instantaneous pitch chroma
        v = hFeatureFunc(X,f_s)
    
    if isTemporal(cFeatureName):
        [v,t] = hFeatureFunc(afAudioData, iBlockLength, iHopLength, f_s)
        
    return (v,t)
    
def isSpectral(cName):
    bResult = False
    if "Spectral" in cName: 
        bResult = True
        
    return (bResult) 

def isTemporal(cName):
    bResult = False
    if "Time" in cName: 
        bResult = True
        
    return (bResult)
    
def computeFeatureCl(cPath, cFeatureName, bPlotOutput):
    from ToolReadAudio import ToolReadAudio
    
    [f_s,afAudioData] = ToolReadAudio(cPath)
    #afAudioData = np.sin(2*np.pi * np.arange(f_s*1)*440./f_s)
 
    [v,t] = computeFeature(cFeatureName, afAudioData, f_s)

    if bPlotOutput:
        plt.plot(t,v)
        
    return (v,t)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute key of wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')
    parser.add_argument('--featurename', metavar='string', required=False,
                        help='feature name in the format SpectralPitchChroma')
    parser.add_argument('--plotoutput', metavar='bool', required=False,
                        help='option to plot the output')
    
    cPath = parser.parse_args().infile
    cFeatureName = parser.parse_args().featurename
    bPlotOutput  = parser.parse_args().plotoutput
    
    #only for debugging
    if not cPath:
        cPath = "c:/temp/test.wav"
    #only for debugging
    if not cFeatureName:
        cFeatureName = "SpectralDecrease"
    if not bPlotOutput:
        bPlotOutput = True
    
    # call the function
    computeFeatureCl(cPath, cFeatureName, bPlotOutput)

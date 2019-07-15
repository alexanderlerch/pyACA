# -*- coding: utf-8 -*-
"""
computes the standard deviation of a time domain signal

  Args:
    x: audio signal
    iBlockLength: block length in samples
    iHopLength: hop length in samples
    f_s: sample rate of audio data (unused)

  Returns:
    vstd standard deviation
    t time stamp
    
"""

import numpy as np
import math

    
def FeatureTimeStd(x, iBlockLength, iHopLength, f_s):   
    
    # number of results
    iNumOfBlocks = math.ceil (x.size/iHopLength)
    
    # compute time stamps
    t = (np.arange(0,iNumOfBlocks) * iHopLength + (iBlockLength/2))/f_s
    
    # allocate memory
    vstd = np.zeros(iNumOfBlocks)
    
    for n in range(0,iNumOfBlocks):
        
        i_start = n*iHopLength
        i_stop  = np.min([x.size-1, i_start + iBlockLength - 1])
        
        # calculate the rms
        vstd[n] = np.std(x[np.arange(i_start,i_stop+1)])
    
    return (vstd,t)
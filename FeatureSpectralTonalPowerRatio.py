# -*- coding: utf-8 -*-
"""
computes the tonal power ratio from the magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data
    G_T: energy threshold

  Returns:
    vtpr tonal power ratio
"""

import numpy as np
from scipy.signal import find_peaks  
    
def FeatureSpectralTonalPowerRatio(X,f_s, G_T=5e-4):   
    
    X = X**2
 
    fSum = X.sum(axis=0)
    vtpr = np.zeros(fSum.shape)

    for n in range(0, X.shape[1]):
        if fSum[n] < G_T:
            continue
        
        # find local maxima above the threshold
        afPeaks = find_peaks(X[:,n], height = G_T)
  
        if not afPeaks[0].size:
            continue
      
        # calculate ratio
        vtpr[n] = X[afPeaks[0],n].sum()/fSum[n]
    
    return (vtpr)


#    % allocate memory
#    vtpr    = zeros(1,size(X,2));
#
#    X       = X.^2;
#    fSum    = sum(X,1);
# 
#    for (n = 1:size(X,2))
#        if (fSum(n) == 0)
#            % do nothing for 0-blocks
#            continue;
#        end
#        % find local maxima
#        [afPeaks]   = findpeaks(X(:,n));
#        
#        % find peaks above the threshold
#        k_peak      = find(afPeaks > G_T);
#        
#        % calculate the ratio
#        vtpr(n)     = sum(afPeaks(k_peak))/fSum(n);
#    end

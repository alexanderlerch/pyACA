# -*- coding: utf-8 -*-

import numpy as np
import math


## helper function: convert Hz to Mel scale
#
#    @param fInHz: frequency to be converted, can be scalar or vector
#    @param cModel: name of the model ('Fant' [default], 'Shaughnessy', 'Umesh')
#
#    @return fMel: Mel values
def ToolFreq2Mel(fInHz, cModel='Fant'):
    # Fant
    def acaFant_scalar(f):
        return 1000 * math.log2(1 + f/1000)
        
    # Shaughnessy
    def acaShaughnessy_scalar(f):
        return 2595 * math.log10(1 + f/700)
        
    # Umesh
    def acaUmesh_scalar(f):
        return f/(2.4e-4*f + 0.741)

    f = np.asarray(fInHz)
    if f.ndim == 0:
        if cModel == 'Shaughnessy':
            return acaShaughnessy_scalar(f)
        elif cModel == 'Umesh':
            return acaUmesh_scalar(f)
        else:
            return acaFant_scalar(f)

    fMel = np.zeros(f.shape)
    if cModel == 'Shaughnessy':
        for k, fi in enumerate(f):
            fMel[k] = acaShaughnessy_scalar(fi)
    elif cModel == 'Umesh':
        for k, fi in enumerate(f):
            fMel[k] = acaUmesh_scalar(fi)
    else:
        for k, fi in enumerate(f):
            fMel[k] = acaFant_scalar(fi)
            
    return fMel

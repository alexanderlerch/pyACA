
import numpy as np
from scipy.signal import lfilter


# see function mfcc.m from Slaneys Auditory Toolbox (Matlab)
def ToolGammatoneFb(afAudioData, f_s, iNumBands=20, f_low=100):

    # initialization
    fEarQ = 9.26449
    fBW = 24.7
    iOrder = 1
    T = 1 / f_s

    # allocate output memory
    X = np.zeros([iNumBands, afAudioData.shape[0]])

    # compute the mid frequencies
    f_c = getMidFrequencies(f_low, f_s / 2, iNumBands, fEarQ, fBW)

    # compute the coefficients
    [afCoeffB, afCoeffA] = getCoeffs(f_c, 1.019 * 2 * np.pi * (((f_c / fEarQ)**iOrder + fBW**iOrder)**(1 / iOrder)), T)

    # do the (cascaded) filter process
    for k in range(0, iNumBands):
        X[k, :] = afAudioData
        for j in range(0, 4):
            X[k, :] = lfilter(afCoeffB[j, :, k], afCoeffA[j, :, k], X[k, :])

    return (X)


# see function ERBSpace.m from Slaneys Auditory Toolbox
def getMidFrequencies(f_low, f_hi, iNumBands, fEarQ, fBW):

    freq = np.log((f_low + fEarQ * fBW) / (f_hi + fEarQ * fBW)) / iNumBands
    f_c = np.exp(np.arange(1, iNumBands + 1) * freq)
    f_c = -(fEarQ * fBW) + f_c * (f_hi + fEarQ * fBW)

    return (f_c)


# see function MakeERBFilters.m from Slaneys Auditory Toolbox
def getCoeffs(f_c, B, T):

    fCos = np.cos(2 * f_c * np.pi * T)
    fSin = np.sin(2 * f_c * np.pi * T)
    fExp = np.exp(B * T)
    fSqrtA = 2 * np.sqrt(3 + 2**(3 / 2))
    fSqrtS = 2 * np.sqrt(3 - 2**(3 / 2))

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * fCos / fExp
    B2 = np.exp(-2 * B * T)

    A11 = -(2 * T * fCos / fExp + fSqrtA * T * fSin / fExp) / 2
    A12 = -(2 * T * fCos / fExp - fSqrtA * T * fSin / fExp) / 2
    A13 = -(2 * T * fCos / fExp + fSqrtS * T * fSin / fExp) / 2
    A14 = -(2 * T * fCos / fExp - fSqrtS * T * fSin / fExp) / 2

    fSqrtA = np.sqrt(3 + 2**(3 / 2))
    fSqrtS = np.sqrt(3 - 2**(3 / 2))
    fArg = (f_c * np.pi * T) * 1j

    fExp1 = 2 * np.exp(4 * fArg)
    fExp2 = 2 * np.exp(-(B * T) + 2 * fArg)

    afGain = np.abs((-fExp1 * T + fExp2 * T * (fCos - fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos - fSqrtA * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtA * fSin)) /
                    (-2 / np.exp(2 * B * T) - fExp1 + (2 + fExp1) / fExp)**4)

    # this is Slaney's compact format - now resort into 3D Matrices
    # fcoefs = [A0*ones(length(f_c),1) A11 A12 A13 A14 A2*ones(length(f_c),1) B0*ones(length(f_c),1) B1 B2 afGain];

    afCoeffB = np.zeros([4, 3, B.size])
    afCoeffA = np.zeros([4, 3, B.size])

    for k in range(0, B.size):
        afCoeffB[0, :, k] = [A0, A11[k], A2] / afGain[k]
        afCoeffA[0, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[1, :, k] = [A0, A12[k], A2]
        afCoeffA[1, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[2, :, k] = [A0, A13[k], A2]
        afCoeffA[2, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[3, :, k] = [A0, A14[k], A2]
        afCoeffA[3, :, k] = [B0, B1[k], B2[k]]

    return (afCoeffB, afCoeffA)


#############################################################################
#    # initialization
#    f_start         = 133.3333
#
#    iNumLinFilters  = 13
#    iNumLogFilters  = 27
#    iNumFilters     = iNumLinFilters + iNumLogFilters
#
#    linearSpacing   = 66.66666666
#    logSpacing      = 1.0711703
#
#    # compute band frequencies
#    f = np.zeros(iNumFilters+2)
#    f[np.arange(0,iNumLinFilters)] = f_start + np.arange(0,iNumLinFilters)*linearSpacing
#    f[np.arange(iNumLinFilters,iNumFilters+2)] = f[iNumLinFilters-1] * logSpacing**np.arange(1,iNumLogFilters+3)
#
#    # sanity check
#    if f[iNumLinFilters-1]>=f_s/2:
#        f = f[f<f_s/2]
#        iNumFilters = f.shape[0] - 2
#
#    f_l = f[np.arange(0,iNumFilters)]
#    f_c = f[np.arange(1,iNumFilters+1)]
#    f_u = f[np.arange(2,iNumFilters+2)]
#
#    # allocate memory for filters and set max amplitude
#    H = np.zeros([iNumFilters,iFftLength])
#    afFilterMax = 2 / (f_u - f_l)
#    f_k = np.arange(0,iFftLength)/(iFftLength-1)*f_s/2
#
#    # compute the transfer functions
#    for c in range(0,iNumFilters):
#        #lower filter slope
#        i_l = np.argmax(f_k>f_l[c])
#        i_u = np.max([0, np.argmin(f_k <= f_c[c])-1])
#        H[c,np.arange(i_l, i_u+1)] = afFilterMax[c] * (f_k[np.arange(i_l, i_u+1)]-f_l[c])/(f_c[c]-f_l[c])
#        #upper filter slope
#        i_l = i_u + 1
#        i_u = np.max([0, np.argmin(f_k < f_u[c])-1])
#        H[c,np.arange(i_l, i_u+1)] = afFilterMax[c] * (f_u[c]-f_k[np.arange(i_l, i_u+1)])/(f_u[c]-f_c[c])
#
#    return (H)

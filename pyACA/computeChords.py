# -*- coding: utf-8 -*-

import numpy as np

from pyACA.computeFeature import computeFeature
from pyACA.ToolPreprocAudio import ToolPreprocAudio
from pyACA.ToolViterbi import ToolViterbi


## recognizes the chords in an audio file
#
#    @param x: array with floating point audio data (dimension samples x channels)
#    @param f_s: sample rate of audio data
#    @param iBlockLength: internal block length (default: 8192 samples)
#    @param iHopLength: internal hop length (default: 2048 samples)
#
#    @return cChordLabel: detected chords as strings
#    @return aiChordIdx: detected chords as indices (2 x iNumObservations)
#    @return t: time stamps
#    @return P_E: full matrix of chord probabilities (iNumChords x iNumObservations)
def computeChords(x, f_s, iBlockLength=8192, iHopLength=2048):

    # chord names
    cChords = ['C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj',
               'F# Maj', 'G Maj', 'G# Maj', 'A Maj', 'A# Maj', 'B Maj',
               'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min',
               'f# min', 'g min', 'g# min', 'a min', 'a# min', 'b min']

    # chord templates
    T = generateTemplateMatrix_I()
    
    # transition probabilities
    P_T = getChordTransProb_I()

    # pre-processing
    x = ToolPreprocAudio(x, iBlockLength)

    # extract pitch chroma
    v_pc, t = computeFeature('SpectralPitchChroma', x, f_s, None, iBlockLength, iHopLength)

    # estimate chord probabilities
    P_E = np.matmul(T, v_pc)
    P_E = P_E / np.sum(P_E, axis=0)

    # allocate space for two rows of results (one raw, one with Viterbi)
    # assign series of labels/indices starting with 0
    aiChordIdx = np.zeros([2, len(t)]).astype(int)
    aiChordIdx[0, :] = np.argmax(P_E, axis=0).astype(int)

    # compute path with Viterbi algorithm
    aiChordIdx[1, :], P_res = ToolViterbi(P_E, P_T, np.ones(len(cChords)) / len(cChords), True)

    # assign result string
    cChordLabel = [[cChords[i] for i in aiChordIdx[0, :]], [cChords[i] for i in aiChordIdx[1, :]]]

    return cChordLabel, aiChordIdx, t, P_E


def generateTemplateMatrix_I():
 
    iNumRootNotes = 12

    # init: 12 major and 12 minor triads
    T = np.zeros([24, 12])
    
    # all chord pitches are weighted equally
    T[0, np.array([0, 4, 7])] = 1/3.
    T[iNumRootNotes, np.array([0, 3, 7])] = 1/3.
    
    # generate templates for all root notes
    for i in range(1, iNumRootNotes):
        T[i, :] = np.roll(T[0, :], i)
        T[i+iNumRootNotes, :] = np.roll(T[iNumRootNotes, :], i)

    return T


def getChordTransProb_I():

    iNumRootNotes = 12
     
    # circle of fifth tonic distances
    circ = np.array([0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5, -3, 4, -1, 6, 1, -4, 3, -2, 5, 0, -5, 2])
        
    # set the circle radius and distance
    R = 1
    d = .5

    # generate key coordinates (mode in z)
    x = R * np.cos(2 * np.pi * circ/float(iNumRootNotes))
    y = R * np.sin(2 * np.pi * circ/float(iNumRootNotes))
    z = np.zeros(2*iNumRootNotes)
    z[0:iNumRootNotes] = d

    P_T = np.zeros([len(x), len(x)])
 
    # compute key distances
    for m in range(len(x)):
        for n in range(len(x)):
            P_T[m, n] = np.sqrt((x[m]-x[n])**2 + (y[m]-y[n])**2 + (z[m]-z[n])**2)

    # convert distances into 'probabilities'
    P_T = .1+P_T
    P_T = 1 - P_T/(.1 + np.max(P_T))
    P_T = P_T / np.sum(P_T, axis=0)

    return P_T



#######################################################
# main
def computeChordsCl(cPath):
    from pyACA.ToolReadAudio import ToolReadAudio

    # read audio file
    [f_s, x] = ToolReadAudio(cPath)
    
    # compute fingerprint
    [cChordLabel, aiChordIdx, t, P_E] = computeChords(x, f_s)

    return cChordLabel, aiChordIdx, t, P_E


if __name__ == "__main__":
    import argparse

    # add command line args and parse them
    parser = argparse.ArgumentParser(description='Compute chords from wav file')
    parser.add_argument('--infile', metavar='path', required=False,
                        help='path to input audio file')

    # retrieve command line args
    args = parser.parse_args()
    cPath = args.infile

    # only for debugging
    if __debug__:
        if not cPath:
            cPath = "../ACA-Plots/audio/sax_example.wav"

    # call the function
    computeChordsCl(cPath)

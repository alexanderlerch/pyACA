# -*- coding: utf-8 -*-

import numpy as np


## helper function: viterbi algorithm
#
#    @param  P_E: emmission probability matrix (states X observations)
#    @param P_T: transition probability matrix (states X states)
#    @param p_s: start probability vector (states X 1)
#    @param bUseLogLikelihood: flag (default: false)
#
#    @return p: path with matrix row indices (length: observations)
#    @return P_res probability matrix
def ToolViterbi(P_E, P_T, p_s, bUseLogLikelihood=False):

    if not bUseLogLikelihood:
        # initialization
        I = np.zeros(P_E.shape).astype(int)
        P_res = np.zeros(P_E.shape)

        P_res[:, 0] = P_E[:, 0] * p_s

        # recursion
        for n in np.arange(1, P_E.shape[1]):
            for s in range(P_E.shape[0]):
                # find max of preceding times trans prob
                p_max = np.max(P_res[:, n-1] * P_T[:, s])
                I[s, n] = np.argmax(P_res[:, n-1] * P_T[:, s]).astype(int)
                P_res[s, n] = P_E[s, n] * p_max
    else:
        # initialization
        P_E = np.log(P_E)  # hope for non-zero entries
        P_T = np.log(P_T)  # hope for non-zero entries
        p_s = np.log(p_s)  # hope for non-zero entries
        I = np.zeros(P_E.shape).astype(int)
        P_res = np.zeros(P_E.shape)

        P_res[:, 0] = P_E[:, 0] + p_s

        # recursion
        for n in np.arange(1, P_E.shape[1]):
            for s in range(P_E.shape[0]):
                # find max of preceding times trans prob
                p_max = np.max(P_res[:, n-1] + P_T[:, s])
                I[s, n] = np.argmax(P_res[:, n-1] + P_T[:, s]).astype(int)
                P_res[s, n] = P_E[s, n] + p_max

    # traceback
    p = np.zeros(P_E.shape[1]).astype(int)
    # start with the last element, then count down
    p[-1] = np.argmax(P_res[:, -1]).astype(int)
    for n in range(P_E.shape[1]-2, -1, -1):
        p[n] = I[p[n+1], n+1]

    return p, P_res

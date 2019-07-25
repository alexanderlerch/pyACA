# -*- coding: utf-8 -*-
"""
helper function: computes path through a distance matrix with simple Dynamic Time Warping

  Args:
    D: non-negative distance matrix

  Returns:
    p: path with matrix indices
    C: cost matrix
"""
import numpy as np


def ToolSimpleDtw(D):

    # init directions for back-tracking
    iDec = np.array([[-1, -1], [-1, 0], [0, -1]])  # DeltaP contents: diag, vert, hori

    # cost initialization
    C = np.zeros(D.shape)
    C[0, :] = np.cumsum(D[0, :])
    C[:, 0] = np.cumsum(D[:, 0])

    # traceback initialization
    DeltaP = np.zeros(D.shape, dtype=int)
    DeltaP[0, :] = 2  # (0,-1)
    DeltaP[:, 0] = 1  # (-1,0)
    DeltaP[0, 0] = 0  # (-1,-1)

    # recursion
    for n_A in range(1, D.shape[0]):
        for n_B in range(1, D.shape[1]):
            # find preceding min (diag, column, row)
            DeltaP[n_A, n_B] = int(np.argmin([C[n_A - 1, n_B - 1], C[n_A - 1, n_B], C[n_A, n_B - 1]]))
            prevC_index = [n_A, n_B] + iDec[DeltaP[n_A, n_B], :]
            C[n_A, n_B] = D[n_A, n_B] + C[prevC_index[0], prevC_index[1]]

    # traceback init
    p = np.asarray(D.shape, dtype=int) - 1  # start with the last element
    n = p

    while (n[0] >= 0) or (n[1] >= 0):
        n = n + iDec[DeltaP[n[0], n[1]], :]

        # update path
        tmp = np.vstack([n, p])
        p = tmp

    return (p[1:, :], C)

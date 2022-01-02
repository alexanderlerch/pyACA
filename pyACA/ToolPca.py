# -*- coding: utf-8 -*-
"""
helper function: principal component analysis (pca)

  Args:
    V: feature matrix (dimension iNumFeatures x iNumObservations)

  Returns:
    U_pc transformed features 'score' (dimension see V)
    T transformation matrix 'loading' (dimension iNumFeatures x iNumFeatures )
    eigenvalues 'latent' (length iNumFeatures)
"""
import numpy as np


def ToolPca(V):

    # covariance
    cov_VV = np.cov(V)

    # svd
    [U, eigenvalues, T] = np.linalg.svd(cov_VV)

    # features to components
    U_pc = np.matmul(T.T, V)

    return U_pc, T, eigenvalues

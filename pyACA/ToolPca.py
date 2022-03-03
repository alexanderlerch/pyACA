# -*- coding: utf-8 -*-

import numpy as np


## helper function: principal component analysis (pca)
#
#    @param V: features for all train observations (dimension iNumFeatures x iNumObservations)
#
#    @return U_pc: transformed features 'score' (dimension see V)
#    @return T: transformation matrix 'loading' (dimension iNumFeatures x iNumFeatures )
#    @return eigenvalues: 'latent' (length iNumFeatures)
def ToolPca(V):

    # covariance
    cov_VV = np.cov(V)

    # svd
    [U, eigenvalues, T] = np.linalg.svd(cov_VV)

    # features to components
    U_pc = np.matmul(T.T, V)

    return U_pc, T, eigenvalues

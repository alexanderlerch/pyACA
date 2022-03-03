# -*- coding: utf-8 -*-

import numpy as np


## helper function: non-negative matrix factorization
#
#    @param X: non-negative matrix to factorize (dimension FFTLength X Observations)
#    @param iRank: nmf rank
#    @param numMaxIter: maximum number of iterations (stop if not converged before, default: 300)
#    @param fSparsity: sparsity weight (default: 0)
#
#    @return W: dictionary matrix
#    @return H: activation matrix
#    @return err: loss function result
def ToolSimpleNmf(X, iRank, numMaxIter=300, fSparsity=0):

    # avoid zero input
    X = X + 1e-30

    # initialization
    [iFreq, iFrames] = X.shape
    err = np.zeros(numMaxIter)
    bUpdateW = True
    bUpdateH = True

    W = np.random.rand(iFreq, iRank)
    H = np.random.rand(iRank, iFrames)

    # normalize W / H matrix
    for r in range(iRank):
        W[:, r] = W[:, r] / np.linalg.norm(W[:, r], 1)

    count = 0
    rep = np.ones([iFreq, iFrames])

    # iteration
    for count in range(numMaxIter):
    
        # current estimate
        X_hat = np.matmul(W, H)
 
        # update
        if bUpdateH:
            H = H * np.matmul(W.T, (X / X_hat)) / np.matmul(W.T, rep)
        if bUpdateW:
            W = W * np.matmul((X / X_hat), H.T) / np.matmul(rep, H.T)
    
        # normalize
        for r in range(iRank):
            W[:, r] = W[:, r] / np.linalg.norm(W[:, r], 1)
       
        # calculate variation between iterations
        err[count] = KlDivergence_I(X, np.matmul(W, H)) + fSparsity * np.linalg.norm(H, 1)

        if count >= 1:
            if (np.abs(err[count] - err[count - 1]) / (err[0] - err[count] + 1e-30)) < 0.001:
                break
        count = count + 1

    return W, H, err[0:count]


def KlDivergence_I(p, q):
    return np.sum(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)) - p + q))

# -*- coding: utf-8 -*-

import numpy as np


class CKMeansState:
    def __init__(self, mean):
        self.mu = mean


## helper function: kmeans clustering
#
#    @param V: features for all train observations (dimension iNumFeatures x iNumObservations)
#    @param K: number of clusters
#    @param numMaxIter: maximum number of iterations (stop if not converged before, default: 1000)
#    @param prevState: internal state that can be stored to continue clustering later
#
#    @return clusterIdx: cluster index of each observation (iNumObservations)
#    @return state: result containing internal state (if needed)
def ToolSimpleKmeans(V, K, numMaxIter=1000, prevState=None):

    # init
    if prevState is None:
        state = CKMeansState(V[:, np.round(np.random.rand(K) * (V.shape[1]-1)).astype(int)])
    else:
        state = CKMeansState(prevState.mu.copy())
    range_V = np.array([np.min(V, axis=1), np.max(V, axis=1)])

    # assign observations to clusters
    clusterIdx = assignClusterLabels_I(V, state)

    for j in range(numMaxIter):
        prevState = CKMeansState(state.mu.copy())

        # update clusters
        state = computeClusterMeans_I(V, clusterIdx, K)

        # reinit empty clusters
        state = reinitState_I(state, clusterIdx, K, range_V)

        # assign observations to clusters
        clusterIdx = assignClusterLabels_I(V, state)
   
        # if converged, break
        if np.max(np.sum(np.abs(state.mu-prevState.mu))) == 0:
            break

    return clusterIdx, state.mu


def assignClusterLabels_I(V, state):

    # number of clusters
    K = state.mu.shape[1]

    D = np.zeros([K, V.shape[1]])
 
    for k in range(K):
        D[k, :] = np.sqrt(np.sum((np.tile(state.mu[:, [k]], (1, V.shape[1])) - V)**2, axis=0, keepdims=True))

    clusterIdx = np.argmin(D, axis=0).astype(int)
    
    return clusterIdx


def computeClusterMeans_I(V, clusterIdx, K):

    # init
    mu = np.zeros([V.shape[0], K])
 
    for k in range(K):
        if np.count_nonzero(clusterIdx == k) != 0:
            mu[:, k] = np.sum(V[:, clusterIdx == k], axis=1) / np.count_nonzero(clusterIdx == k)
     
    return CKMeansState(mu)


def reinitState_I(state, clusterIdx, K, range_V):

    for k in range(K):
        if np.count_nonzero(clusterIdx == k) == 0:
            state.mu[:, k] = np.random.rand(state.mu.shape[0], 1) * (range_V[:, 1]-range_V[:, 0]) + range_V[:, 0]

    return state

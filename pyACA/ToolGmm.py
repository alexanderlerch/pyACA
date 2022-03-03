# -*- coding: utf-8 -*-

import numpy as np


class CGmmState:
    def __init__(self, mean, sigma, prior):
        self.mu = mean
        self.sigma = sigma
        self.prior = prior


## helper function: gaussian mixture model
#
#    @param V: features for all train observations (dimension iNumFeatures x iNumObservations)
#    @param K: number of gaussians
#    @param numMaxIter: maximum number of iterations (stop if not converged before, default: 1000)
#    @param prevState: internal state that can be stored to continue clustering later
#
#    @return mu: means (iNumFeatures x K)
#    @return sigma: standard deviations (K x iNumFeatures X iNumFeatures)
#    @return state: result containing internal state (if needed)
def ToolGmm(V, K, numMaxIter=1000, prevState=None):

    # init
    if prevState is None:
        state = initState_I(V, K)
    else:
        state = CGmmState(prevState.mu.copy(), prevState.sigma.copy(), prevState.prior.copy())

    for j in range(numMaxIter):
        prevState = CGmmState(state.mu.copy(), state.sigma.copy(), state.prior.copy())

        # compute weighted gaussian
        p = computeProb_I(V, state)

        # update clusters
        state = updateGaussians_I(V, p, state)
   
        # if converged, break
        if np.max(np.sum(np.abs(state.mu-prevState.mu))) <= 1e-20:
            break

    return state.mu, state.sigma, state


def updateGaussians_I(V, p, state):

    # number of clusters
    K = state.mu.shape[1]
 
    # update priors
    state.prior = np.mean(p, axis=0)

    for k in range(K):
        s = 0

        # update means
        state.mu[:, k] = np.matmul(V, p[:, k]) / np.sum(p[:, k])
        
        # subtract mean
        Vm = V - state.mu[:, [k]]
        
        for n in range(V.shape[1]):
            s = s + p[n, k] * np.matmul(Vm[:, [n]], Vm[:, [n]].T)

        state.sigma[k, :, :] = s / np.sum(p[:, k])
    
    return state


def computeProb_I(V, state):

    K = state.mu.shape[1]
    p = np.zeros([V.shape[1], K])
    
    # for each cluster
    for k in range(K):
        # subtract mean
        Vm = V - state.mu[:, [k]]

        # weighted gaussian
        p[:, k] = 1 / np.sqrt((2*np.pi)**Vm.shape[0] * np.linalg.det(state.sigma[k, :, :])) * np.exp(-.5 * np.sum(np.multiply(Vm, np.matmul(np.linalg.inv(state.sigma[k, :, :]), Vm)), axis=0).T)
        p[:, k] = state.prior[k] * p[:, k]

    # norm over clusters
    p = p / np.tile(np.sum(p, axis=1, keepdims=True), (1, K))

    return p

 
def initState_I(V, K):

    prior = np.ones(K) / K

    # pick random points as cluster means
    mIdx = np.round(np.random.rand(K) * (V.shape[1]-1)).astype(int)
 
    # assign means etc.
    mu = V[:, mIdx]
    s = np.cov(V)
    sigma = np.zeros([K, V.shape[0], V.shape[0]])
    for k in range(K):
        sigma[k, :, :] = s

    # write initial state
    state = CGmmState(mu, sigma, prior)

    return state

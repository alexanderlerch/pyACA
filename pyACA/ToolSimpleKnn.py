# -*- coding: utf-8 -*-

import numpy as np


## helper function: k nearest neighbor classifier
#
#    @param v_test: test feature vector
#    @param V_train: features for all train observations (dimension iNumFeatures x iNumObservations)
#    @param ClassIdx_train: class labels (length observations)
#    @param K: number of neighbors taken into account (default = 3)
#
#    @return est_class: index of estimated class
def ToolSimpleKnn(v_test, V_train, ClassIdx_train, K=1):

    # sanity checks
    if v_test.shape[0] != V_train.shape[0]:
        return -1

    # get dimensions
    # num_features = TestFeatureVector.shape[0]
    classes = getClasses_I(ClassIdx_train)

    # init result
    est_class = -1*np.ones(v_test.shape[1])

    # compute pairwise distances between all test data and all train data points
    d = computePairwiseDistance_I(v_test, V_train)

    # sort distances
    ind = np.argsort(d, axis=1).astype(int)
    ind = ind[:, range(K)]

    # extension for distance based weighting: convert distance to closeness (easier later with the histogram)
    ma = np.amax(d)
    if ma <= 0:
        ma = 1
    d = 1-d/ma

    # infer which class
    for obs, index in enumerate(ind):
        est_class[obs] = inferClass_I(ClassIdx_train[index], classes, d[obs, index])

    return np.squeeze(est_class.astype(int))


def getClasses_I(labels):
    
    return np.unique(labels)


def computePairwiseDistance_I(test_data, train_data):
    
    # you may also use sp.spatial.distance.cdist
    d = np.zeros((test_data.shape[1], train_data.shape[1]))
    for i, v in enumerate(test_data.T):
        d[i, :] = computeEucDist_I(v, train_data.T)
    return d


def computeEucDist_I(vector, matrix):
    
    d = np.sum((vector - matrix)**2, axis=1, keepdims=True)
    return np.sqrt(d.T)


def inferClass_I(closest_labels, classes, closeness):
    
    # first, try to do a simple majority vote
    # res_class = sp.stats.mode(class_labels, axis = 0)
    hist = np.histogram(closest_labels, bins=len(classes), range=(min(classes), max(classes)))
 
    # check if we have a clear majority
    s = np.flip(np.sort(hist[0]))
    # fallback: if not, do a histogram weighted with the closeness (inverted distance)
    if s[0] == s[1]:
        hist = np.histogram(closest_labels, bins=len(classes), range=(min(classes), max(classes)), weights=closeness)
 
    return classes[np.argmax(hist[0])]

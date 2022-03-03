# -*- coding: utf-8 -*-

import numpy as np

from pyACA import ToolLooCrossVal


## helper function: sequential forward feature selection
#
#    @param V: feature matrix with all observations (dimension iNumFeatures x iNumObservations)
#    @param ClassIdx: class labels (length: iNumObservations)
#    @param iNumFeatures2Select: target number of features (default: all)
#
#    @return selFeatureIdx: vector with ordered feature indices (length: iNumFeatures2Select)
#    @return AccPerSubset: accuracy for each subset
def ToolSeqFeatureSel(V, ClassIdx, iNumFeatures2Select=-1):

    iNumFeatures = V.shape[0]
    if iNumFeatures2Select <= 0 or iNumFeatures2Select > iNumFeatures:
        iNumFeatures2Select = iNumFeatures

    # initialize
    selFeatureIdx = -1 * np.ones(iNumFeatures2Select).astype(int)
    unselFeatures = np.ones(iNumFeatures).astype(bool)
    AccPerSubset = np.zeros(iNumFeatures2Select)
    accTmp = np.zeros(iNumFeatures)

    # find single best feature
    for f in range(iNumFeatures):
        # get accuracy of selected features plus current feature f
        accTmp[f], fold_accuracies, conf_mat = ToolLooCrossVal(V[f, :][None, :], ClassIdx)
    selFeatureIdx[0] = np.argmax(accTmp, axis=0).astype(int)
    unselFeatures[selFeatureIdx[0]] = False
    AccPerSubset[0] = accTmp[selFeatureIdx[0]]

    # iterate until target number of features is reached
    for i in np.arange(1, iNumFeatures2Select):
        # iterate over all features not yet selected
        for f in range(iNumFeatures):
            if unselFeatures[f]:
                # get accuracy of selected features plus current feature f
                subset = np.zeros(i+1).astype(int)
                subset[:i] = selFeatureIdx[:i].astype(int)
                subset[i] = int(f)
                accTmp[f], fold_accuracies, conf_mat = ToolLooCrossVal(V[subset, :], ClassIdx)
            else:
                accTmp[f] = -1
                continue
        
        # identify feature maximizing the accuracy
        # move feature from unselected to selected
        selFeatureIdx[i] = np.argmax(accTmp, axis=0).astype(int)
        unselFeatures[selFeatureIdx[i]] = False
        AccPerSubset[i] = accTmp[selFeatureIdx[i]]

    return selFeatureIdx, AccPerSubset

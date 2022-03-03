# -*- coding: utf-8 -*-

import numpy as np

from pyACA.ToolSimpleKnn import ToolSimpleKnn


## helper function: leave one out cross validation
#
#    @param V: feature matrix with all observations (dimension iNumFeatures x iNumObservations)
#    @param ClassIdx: class labels (length: iNumObservations)
#
#    @return avg_accuracy: overall accuracy
#    @return fold_accuracies: accuracies per fold
#    @return conf_mat: confusion matrix
def ToolLooCrossVal(V, ClassIdx):

    if V.ndim == 1:
        V = V[None, :]

    iNumObservations = V.shape[1]
    kNearestNeighbor = 3

    return crossvalidate_I(V, ClassIdx, iNumObservations, kNearestNeighbor)


def crossvalidate_I(data, labels, num_folds=10, k=3):

    # init result
    classes = getClasses_I(labels)
    fold_accuracies = np.zeros(num_folds)
    conf_mat = np.zeros((num_folds, len(classes), len(classes))) 

    # split data
    fold_ind = splitData_I(labels, num_folds, classes)

    # do classification for each fold
    for n in range(num_folds):
        train_data = np.zeros((data.shape[0], 0))
        test_data = np.zeros((data.shape[0], 0))
        train_label = np.zeros(0)
        
        # split train and testdata
        for i, train in enumerate(fold_ind):
            if i == n:
                test_data = np.hstack((test_data, data[:, fold_ind[n]]))
                test_label = labels[np.squeeze(fold_ind[i])].astype(int)
                continue
            train_data = np.hstack((train_data, data[:, fold_ind[i]]))
            train_label = np.append(train_label, labels[fold_ind[i]])

        # classify
        est_label = ToolSimpleKnn(test_data, train_data, train_label, k)

        # evaluate result
        [fold_accuracies[n], conf_mat[n, :, :]] = evaluate_I(test_label, est_label, classes)

    # compute overall metrics from fold results    
    avg_accuracy = np.mean(fold_accuracies)
    conf_mat = np.sum(conf_mat, axis=0)

    return avg_accuracy, fold_accuracies, conf_mat


def getClasses_I(labels):
    
    return np.unique(labels)


def splitData_I(gt_labels, num_folds, classes):

    # check number of observations per class    
    num_obs_class = np.zeros(len(classes))
    for k, c in enumerate(classes):
        num_obs_class[k] = int(len(gt_labels[gt_labels == c]))
    num_obs_class = num_obs_class.astype(int)

    # compute observations per fold stratified
    avg_obs_fold = np.floor(len(gt_labels)/num_folds).astype(int)
    num_obs_fold = np.zeros([num_folds, len(classes)]).astype(int)
    while np.sum(num_obs_class) > np.sum(num_obs_fold) and np.sum(num_obs_fold) <= num_folds * avg_obs_fold:
        for k in range(len(classes)):
            for f in range(num_folds):
                if np.sum(num_obs_fold[f, :]) >= avg_obs_fold:
                    continue
                if num_obs_class[k]-np.sum(num_obs_fold[:, k]) > 0:
                    num_obs_fold[f, k] += 1
                else:
                    break
    
    # take care of any unassigned stragglers
    while np.sum(num_obs_class) > np.sum(num_obs_fold):
        for k in range(len(classes)):
            if num_obs_class[k]-np.sum(num_obs_fold[:, k]) <= 0:
                continue
            for f in range(num_folds):
                num_obs_fold[f, k] += 1
     
    # split actual data into folds
    last = np.zeros(len(classes)).astype(int)
    data_ind = [[] for _ in range(num_folds)]
    for f in range(num_folds):
        for k, c in enumerate(classes):
            num = num_obs_fold[f, k]
            data_ind[f] = data_ind[f] + np.argwhere(gt_labels == c)[np.arange(last[k], last[k]+num)].astype(int).tolist()
            last[k] += num

    for f in range(num_folds):
        data_ind[f] = np.ravel(data_ind[f]).astype(int)

    return data_ind


def evaluate_I(gt, est, class_indices):
    # compute confusion matrix
    conf_mat = computeConfMat_I(gt, est, class_indices)

    gt = np.asarray(gt)
    if gt.ndim == 0:
        accuracy = np.trace(conf_mat)
    else:
        accuracy = np.trace(conf_mat)/len(gt)

    return accuracy, conf_mat


def computeConfMat_I(gt, est, class_indices):
    conf_mat = np.zeros((len(class_indices), len(class_indices)))

    gt = np.asarray(gt)
    if gt.ndim == 0:
        conf_mat[gt - class_indices[0], est - class_indices[0]] += 1
    else:
        # assume the class indices to be increasing integers
        for i, row in enumerate(gt):
            conf_mat[row - class_indices[0], np.asarray(est)[i] - class_indices[0]] += 1

    return conf_mat.astype(int)

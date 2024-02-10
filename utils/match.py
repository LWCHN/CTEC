# -*- coding: utf-8 -*-

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
import numpy as np
import pdb

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    y_voted_labels = np.zeros(y_true.shape)

    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


def result_hungarian_match(preds, targets):
    class_num = len(np.unique(targets))
    num_samples = len(preds)
    num_correct = np.zeros((class_num, class_num))
    for c1 in range(0, class_num):
        for c2 in range(0, class_num):
            votes = int(((preds == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes
    match = linear_assignment(num_samples - num_correct)
    a_ind, b_ind = match
    res = []

    ## for scipy.optimize import linear_sum_assignment
    ## use: from scipy.optimize import linear_sum_assignment as linear_assignment
    for i in range(0,len(a_ind)):
        res.append((a_ind[i], b_ind[i] )) 

    reordered_preds = np.zeros(num_samples)
    for i in range(0,len(a_ind)):
        pred_i = a_ind[i]
        target_i = b_ind[i]
        reordered_preds[preds == pred_i] = int(target_i)

    ari = metrics.adjusted_rand_score(targets, preds) * 100
    nmi = metrics.normalized_mutual_info_score(targets, preds) * 100
    pur = purity_score(targets, reordered_preds) * 100

    acc = np.sum((reordered_preds == targets)) / float(len(targets)) * 100
    f1 = metrics.f1_score(targets, reordered_preds, average='macro') * 100
    pre = metrics.precision_score(targets, reordered_preds, average='macro') * 100
    recall = metrics.recall_score(targets, reordered_preds, average='macro') * 100

    return res,reordered_preds, acc, pre, recall, f1, ari, nmi, pur

def result_hungarian_match_ver1(preds, targets):
    class_num = len(np.unique(targets))
    num_samples = len(preds)
    num_correct = np.zeros((class_num, class_num))
    for c1 in range(0, class_num):
        for c2 in range(0, class_num):
            votes = int(((preds == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes
    match = linear_assignment(num_samples - num_correct)
    res = []
    for out_c, gt_c in match: ## sklearn need 0.22.1
        res.append((out_c, gt_c))

    reordered_preds = np.zeros(num_samples)
    for pred_i, target_i in match:
        reordered_preds[preds == pred_i] = int(target_i)

    ari = metrics.adjusted_rand_score(targets, preds) * 100
    nmi = metrics.normalized_mutual_info_score(targets, preds) * 100
    pur = purity_score(targets, reordered_preds) * 100

    acc = np.sum((reordered_preds == targets)) / float(len(targets)) * 100
    f1 = metrics.f1_score(targets, reordered_preds, average='macro') * 100
    pre = metrics.precision_score(targets, reordered_preds, average='macro') * 100
    recall = metrics.recall_score(targets, reordered_preds, average='macro') * 100

    return res,reordered_preds, acc, pre, recall, f1, ari, nmi, pur
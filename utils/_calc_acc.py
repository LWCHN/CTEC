# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics


# sys.path.append('./')
# from utils import match

def list2str(one_list):
    mystr=''
    for i in range(0,len(one_list)):
        if i==0:
            mystr=str(one_list[i])+','
        elif i==len(one_list)-1:
            mystr = mystr+str(one_list[i])
        else:
            mystr = mystr+str(one_list[i])+','
    return mystr


def gt_str2nbr(list_str):
    label = list_str #list(pd.read_csv(info_path)['celltype'])
    LE = pp.LabelEncoder()
    label = LE.fit_transform(label)
    return np.asarray(list(label))

def calc_all_acc_simple(df,gt_name,pred_name,decimals=2):
    label_str=np.asarray(list(df.loc[:,gt_name]))
    label_nbr = gt_str2nbr(label_str)
    pred = np.asarray(list(df.loc[:,pred_name]))
    ##########res,reordered_preds, acc, pre, recall, f1, ari, nmi, pur = match.result_hungarian_match(pred, label_nbr)
    ari = adjusted_rand_score(label_nbr, pred)
    nmi = normalized_mutual_info_score(label_nbr, pred)
    all_result_value = np.array([ari, nmi])
    all_result_value = list(np.round(all_result_value,decimals=decimals))
    all_result_name = [ 'ARI', 'NMI']
    return all_result_value, all_result_name
    
def evaluate_df(df_input,pred_name,GTname):
    df = df_input[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name

def evaluate(adata,pred_name,GTname):
    df = adata.obs[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name







# def purity_score(y_true, y_pred):
#     """Purity score
#         Args:
#             y_true(np.ndarray): n*1 matrix Ground truth labels
#             y_pred(np.ndarray): n*1 matrix Predicted clusters

#         Returns:
#             float: Purity score
#     """
#     y_voted_labels = np.zeros(y_true.shape)

#     labels = np.unique(y_true)
#     ordered_labels = np.arange(labels.shape[0])
#     for k in range(labels.shape[0]):
#         y_true[y_true==labels[k]] = ordered_labels[k]
#     labels = np.unique(y_true)
#     bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

#     for cluster in np.unique(y_pred):
#         hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
#         winner = np.argmax(hist)
#         y_voted_labels[y_pred==cluster] = winner

#     return metrics.accuracy_score(y_true, y_voted_labels)


# def result_hungarian_match(preds, targets):
#     class_num = len(np.unique(targets))
#     num_samples = len(preds)
#     num_correct = np.zeros((class_num, class_num))
#     for c1 in range(0, class_num):
#         for c2 in range(0, class_num):
#             votes = int(((preds == c1) * (targets == c2)).sum())
#             num_correct[c1, c2] = votes
#     match = linear_assignment(num_samples - num_correct)
#     a_ind, b_ind = match
#     res = []

#     ## for scipy.optimize import linear_sum_assignment
#     ## use: from scipy.optimize import linear_sum_assignment as linear_assignment
#     for i in range(0,len(a_ind)):
#         res.append((a_ind[i], b_ind[i] )) 

#     reordered_preds = np.zeros(num_samples)
#     for i in range(0,len(a_ind)):
#         pred_i = a_ind[i]
#         target_i = b_ind[i]
#         reordered_preds[preds == pred_i] = int(target_i)

#     ari = metrics.adjusted_rand_score(targets, preds) * 100
#     nmi = metrics.normalized_mutual_info_score(targets, preds) * 100
#     pur = purity_score(targets, reordered_preds) * 100

#     acc = np.sum((reordered_preds == targets)) / float(len(targets)) * 100
#     f1 = metrics.f1_score(targets, reordered_preds, average='macro') * 100
#     pre = metrics.precision_score(targets, reordered_preds, average='macro') * 100
#     recall = metrics.recall_score(targets, reordered_preds, average='macro') * 100

#     return res,reordered_preds, acc, pre, recall, f1, ari, nmi, pur

# # def result_hungarian_match_ver1(preds, targets):
# #     class_num = len(np.unique(targets))
# #     num_samples = len(preds)
# #     num_correct = np.zeros((class_num, class_num))
# #     for c1 in range(0, class_num):
# #         for c2 in range(0, class_num):
# #             votes = int(((preds == c1) * (targets == c2)).sum())
# #             num_correct[c1, c2] = votes
# #     match = linear_assignment(num_samples - num_correct)
# #     res = []
# #     for out_c, gt_c in match: ## sklearn need 0.22.1
# #         res.append((out_c, gt_c))

# #     reordered_preds = np.zeros(num_samples)
# #     for pred_i, target_i in match:
# #         reordered_preds[preds == pred_i] = int(target_i)

# #     ari = metrics.adjusted_rand_score(targets, preds) * 100
# #     nmi = metrics.normalized_mutual_info_score(targets, preds) * 100
# #     pur = purity_score(targets, reordered_preds) * 100

# #     acc = np.sum((reordered_preds == targets)) / float(len(targets)) * 100
# #     f1 = metrics.f1_score(targets, reordered_preds, average='macro') * 100
# #     pre = metrics.precision_score(targets, reordered_preds, average='macro') * 100
# #     recall = metrics.recall_score(targets, reordered_preds, average='macro') * 100

# #     return res,reordered_preds, acc, pre, recall, f1, ari, nmi, pur


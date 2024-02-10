# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


sys.path.append('./')
from utils import match

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
    

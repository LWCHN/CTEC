# -*- coding: utf-8 -*-
import re
import os
import scanpy as sc
import numpy as np
import time
import pandas as pd
import sklearn.preprocessing as pp
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


sys.path.append('./')

def evaluate(adata,pred_name,GTname):
    df = adata.obs[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name


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
    ari = adjusted_rand_score(label_nbr, pred)
    nmi = normalized_mutual_info_score(label_nbr, pred)
    all_result_value = np.array([ari, nmi])
    all_result_value = list(np.round(all_result_value,decimals=decimals))
    all_result_name = [ 'ARI', 'NMI']
    return all_result_value, all_result_name
    


def makeDIR(*path):
    for new_path in path:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        assert(os.path.exists(new_path))
    return new_path



def dir_reader(data_dir, filter_pattern=None, sub_dir=None, depth=100):
    assert isinstance(depth, int)
    depth = max(depth - 1, 0)
    full_paths = []
    cur_paths = []
    if depth:
        current_dir = data_dir if not sub_dir else os.path.join(data_dir, sub_dir)
        sub_files = os.listdir(current_dir)
        for sub_file in sub_files:
            sub_file_path = sub_file if not sub_dir else os.path.join(sub_dir, sub_file)
            file_path = os.path.join(current_dir, sub_file)
            
            if os.path.isdir(file_path):
                sub_full_paths, sub_cur_paths = dir_reader(data_dir, filter_pattern, sub_dir=sub_file_path, depth=depth)
                full_paths += sub_full_paths
                cur_paths += sub_cur_paths
            elif not filter_pattern or re.match(filter_pattern, sub_file_path):
                full_paths.append(os.path.join(data_dir, sub_file_path))
                cur_paths.append(sub_file_path)
            else:
                pass
                #print('filter', file_path)
    return full_paths, cur_paths

def win_to_linux(win_path):
    return win_path.replace('\\','/')


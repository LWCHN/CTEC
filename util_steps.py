# -*- coding: utf-8 -*-
"""
"""

from operator import is_not
import os
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ["PYTHONIOENCODING"] = "utf-8"
import scanpy as sc
import numpy as np
import time
import pandas as pd

## import algorithm file
from utils.algo_ctec import pre_process, cluster_leiden,find_resolution_leiden,find_resolution_louvain,evaluate
from utils.algo_ctec import algo_DESC
from utils.algo_ensemble_cv import cell_label_update_CV_df
from utils.algo_ensemble_od import cell_label_update_OD_df
from utils import calc_acc
from utils import path_util
import warnings
warnings.filterwarnings("ignore")
import gc
import argparse

    
def loaddata_KnownCluster(DATASET_NUM):
    ###############################
    if DATASET_NUM == 1:
        data_name = 'macaque_bc'
        data_ext = '.h5ad'
        GTname = 'cluster'
        class_nums = 12
        BATCH_KEY = 'macaque_id'
        N_HIGH_VAR=1000
        res_leiden =0.390625
        res_louvain =0.29296875
    ###############################
    if DATASET_NUM == 2:
        data_name = 'human_pbmc_GSE96583'
        data_ext = '.h5ad'
        GTname = 'cell'
        class_nums = 8
        BATCH_KEY = "stim"
        N_HIGH_VAR=1000
        res_leiden = 0.244140625
        res_louvain = 0.2197265625
    ###############################
    if DATASET_NUM == 3:
        data_name = 'mouse_cortex_SCP425'
        data_ext = '.h5ad'
        GTname = 'CellType'
        class_nums = 8
        BATCH_KEY = 'Method'
        N_HIGH_VAR=1000
        res_leiden = 0.146484375
        res_louvain = 0.146484375
    ###############################    
    if DATASET_NUM == 4:
        data_name = 'human_pancreas'
        data_ext = '.h5ad'
        GTname = 'celltype'
        class_nums = 13
        BATCH_KEY = 'protocol'
        N_HIGH_VAR=1000
        res_leiden = 0.390625
        res_louvain = 0.78125
    ###############################    
    if DATASET_NUM == 5:
        data_name = 'paul15'
        data_ext = '.h5ad'
        GTname = 'paul15_label'
        class_nums = 10
        BATCH_KEY = None
        N_HIGH_VAR=1000
        res_leiden = 0.9765625
        res_louvain = 1.07421875
    return data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain


def loaddata_default(DATASET_NUM):
    ## part 1: read and preprocesing
    ###############################
    if DATASET_NUM == 1:
        data_name = 'macaque_bc'
        data_ext = '.h5ad'
        GTname = 'cluster'
        class_nums = 12
        BATCH_KEY = 'macaque_id'
        N_HIGH_VAR=1000
        res_leiden =1.0
        res_louvain =1.0
    ###############################
    if DATASET_NUM == 2:
        data_name = 'human_pbmc_GSE96583'
        data_ext = '.h5ad'
        GTname = 'cell'
        class_nums = 8
        BATCH_KEY = "stim"
        N_HIGH_VAR=1000
        res_leiden = 1.0
        res_louvain = 1.0
    ###############################
    if DATASET_NUM == 3:
        data_name = 'mouse_cortex_SCP425'
        data_ext = '.h5ad'
        GTname = 'CellType'
        class_nums = 8
        BATCH_KEY = 'Method'
        N_HIGH_VAR= 1000
        res_leiden = 1.0
        res_louvain = 1.0
    ###############################    
    if DATASET_NUM == 4:
        data_name = 'human_pancreas'
        data_ext = '.h5ad'
        GTname = 'celltype'
        class_nums = 13
        BATCH_KEY = 'protocol'
        N_HIGH_VAR=1000
        res_leiden = 1.0
        res_louvain = 1.0
    ###############################    
    if DATASET_NUM == 5:
        data_name = 'paul15'
        data_ext = '.h5ad'
        GTname = 'paul15_label'
        class_nums = 10
        BATCH_KEY = None
        N_HIGH_VAR=1000
        res_leiden = 1.0
        res_louvain = 1.0
    ###############################
    if DATASET_NUM == 6:
        data_name = 'mouse_retina_GSE81904'
        data_ext = '.h5ad'
        GTname = 'celltype'
        class_nums = 14
        BATCH_KEY = 'BatchID'
        N_HIGH_VAR=1000
        res_leiden = 1.0
        res_louvain = 1.0
        
    return data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain

# def loaddata_KnownCluster(DATASET_NUM):
#     ## part 1: read and preprocesing
#     ###############################
#     if DATASET_NUM == 1:
#         data_name = 'macaque_bc'
#         data_ext = '.h5ad'
#         GTname = 'cluster'
#         class_nums = 12
#         BATCH_KEY = 'macaque_id'
#         N_HIGH_VAR=1000
#         res_leiden =0.390625
#         res_louvain =0.29296875
#     ###############################
#     if DATASET_NUM == 2:
#         data_name = 'human_pbmc_GSE96583'
#         data_ext = '.h5ad'
#         GTname = 'cell'
#         class_nums = 8
#         BATCH_KEY = "stim"
#         N_HIGH_VAR=1000
#         res_leiden = 0.244140625
#         res_louvain = 0.2197265625
#     ###############################
#     if DATASET_NUM == 3:
#         data_name = 'mouse_cortex_SCP425'
#         data_ext = '.h5ad'
#         GTname = 'CellType'
#         class_nums = 8
#         BATCH_KEY = 'Method'
#         N_HIGH_VAR=1000
#         res_leiden = 0.146484375
#         res_louvain = 0.146484375
#     ###############################    
#     if DATASET_NUM == 4:
#         data_name = 'human_pancreas'
#         data_ext = '.h5ad'
#         GTname = 'celltype'
#         class_nums = 13
#         BATCH_KEY = 'protocol'
#         N_HIGH_VAR=1000
#         res_leiden = 0.390625
#         res_louvain = 0.78125
#     ###############################    
#     if DATASET_NUM == 5:
#         data_name = 'paul15'
#         data_ext = '.h5ad'
#         GTname = 'paul15_label'
#         class_nums = 10
#         BATCH_KEY = None
#         N_HIGH_VAR=1000
#         res_leiden = 0.9765625
#         res_louvain = 1.07421875
#     return data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain

def save_time_to_txt(path_txt,time_used):
    #w tells python we are opening the file to write into it
    outfile = open(path_txt, 'a+')
    outfile.write(str(time_used)+',')
    outfile.close() #Close the file when we’re done!  


def evaluate_df(df_input,pred_name,GTname):
    df = df_input[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_acc.calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name

def win_to_linux(win_path):
    return win_path.replace('\\','/')
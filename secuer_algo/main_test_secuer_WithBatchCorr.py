# -*- coding: utf-8 -*-
from operator import is_not
import os
import scanpy as sc
import numpy as np
import time
from tools import *
import warnings
warnings.filterwarnings("ignore")
import gc
import argparse


import secuer as sr
def algo_secuer(data):
    fea = data.obsm['X_pca']
    res = sr.secuer(fea= fea,
                    Knn=5,
                    multiProcessState=True,
                    num_multiProcesses=4)

    # # run secuer-consensus
    # resC = sr.secuerconsensus(run_secuer=True,
    #                         fea= fea,
    #                         Knn=5,
    #                         M=5,
    #                         multiProcessState=True,
    #                         num_multiProcesses=4)
    return res


    
def pre_process(adata, N_HIGH_VAR,BATCH_KEY):
    t0=time.time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_combat = adata.copy()
    del adata
    sc.pp.highly_variable_genes(adata_combat, n_top_genes=N_HIGH_VAR, batch_key=BATCH_KEY)
    adata_combat = adata_combat[:, adata_combat.var.highly_variable]
    if not BATCH_KEY == None:
        print('    pre_process ......sc.pp.combat')
        sc.pp.combat(adata_combat, key = BATCH_KEY)
    sc.pp.scale(adata_combat, max_value=10)
    print('    pre_process ......time:',time.time()-t0)
    return adata_combat


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







#######################
DATA_PATH = r'D:\aaa\dataset\paper_DESC_dataset'
WORK_PATH = r'D:\CTEC_submit_work\review_BIOINF\project_algo_review\secuer_algo' 

DATA_PATH  = win_to_linux(DATA_PATH)
WORK_PATH  = win_to_linux(WORK_PATH)

SAVE_PATH = WORK_PATH
secuer_name_pattern = '_processed_SECUER_res_'

parser = argparse.ArgumentParser()
parser.add_argument('--data_nbr', type=int, default=None, required=False, help='ID of dataset')
parser.add_argument('--exp_id', type=str, default='WithBatchCorr', required=False, help='experiment id')
parser.add_argument('--save_h5ad', type=bool, default=False, required=False, help='if save final h5ad result')
args = parser.parse_args()    
EXP = args.exp_id
SAVE_H5AD = args.save_h5ad
if EXP == 0:
    SAVE_H5AD = True


## manage result save path:
save_path = os.path.join(SAVE_PATH,'result_secuer_'+str(EXP))
try:
    os.mkdir(save_path)
except:
    pass
print('~~~~~~~~~~     save_path = ',save_path)


if args.data_nbr is None:
    DATASET_NUM_LIST=[1,2,3,4,5]
else:
    DATASET_NUM_LIST = [args.data_nbr]


pd_dict = dict()
for i in range(0,len(DATASET_NUM_LIST)):
    DATASET_NUM = DATASET_NUM_LIST[i]
    print('\n\n','='*100)
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)


    #[read data meta info]
    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,_,_ = loaddata_default(DATASET_NUM)
    print('data_name = ',data_name,'\n')


    # [wash data]
    try:
        del adata
    except:
        pass
    adata = sc.read(os.path.join(DATA_PATH,data_name+data_ext))
    if DATASET_NUM==5:
        adata.obs['paul15_label'] = adata.obs['paul15_clusters'].str.split(
        "[0-9]{1,2}", n=1, expand=True).values[:, 1]
        adata.obs['paul15_label'].astype('category')
    if not GTname == None:
        adata = adata[adata.obs[GTname].notna()] ###### find the nan from label, and remove them
        
    adata.obs[GTname] = adata.obs[GTname].astype('category')
    adata.obs['celltype'] = adata.obs[GTname]
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    ADATA_RAW = adata.copy()


    # [pre_process data]
    print('    pre_process ......')
    # same pre-process for all paras
    adata_pre = pre_process(adata, N_HIGH_VAR,BATCH_KEY)
    print('adata_pre',adata_pre)
    adata_for_secuer = adata_pre.copy()
    # same PCA for all paras
    sc.tl.pca(adata_for_secuer, svd_solver='arpack',random_state=1) # use default paras
    sc.pp.neighbors(adata_for_secuer, n_neighbors=10, use_rep="X_pca", random_state=1) # use default paras
    print('adata_for_secuer:\n',adata_for_secuer)


    # [use secuer for result]
    print('    secuer method ......')
    t0=time.time()
    res_secuer = algo_secuer(adata_for_secuer) #输出矩阵，只包含类别的int
    time_secuer=round(time.time()-t0,3)
    print("time_secuer  = ",time_secuer )


    # save cluster into adta
    ADATA_RAW.obs['secuer_labels'] = res_secuer.tolist() 
    ADATA_RAW.obs['secuer_labels'] = ADATA_RAW.obs['secuer_labels'].astype('category')
    res_secuer = 999
    ADATA_RAW.obs.to_csv(os.path.join(save_path, data_name+secuer_name_pattern+'.csv'))
    ADATA_RAW.write_h5ad(os.path.join(save_path, data_name+'_cluster_SECUER.h5ad'))


    # [evaluation]
    # try:
    if 1:
        ## [] calc acc
        acc_results_value, acc_results_name = evaluate(ADATA_RAW,'secuer_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_SECUER.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('Acc_Name,'+list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_SECUER,'+list2str(acc_results_value)+','+str(res_secuer)+','+str(time_secuer)+'\n')
            f.write('\n')
    # except:
    #     pass
        acc_algo = pd.read_csv(result_acc_path,delimiter = '\t')
        pd_dict.update({data_name:['method_name','ARI','NMI','resolution','time']})
        pd_dict.update({data_name+'_SECUER':acc_algo.iloc[-1,0].split(',')})
    # except:
    #     pass

# [evaluation]
acc_one_data = pd.DataFrame.from_dict(pd_dict,orient = 'index')
print('acc_one_data = \n',acc_one_data)
acc_one_data.to_csv(os.path.join(save_path,'result_acc====_Make_Acc_Table.csv'))

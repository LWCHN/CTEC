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

import pandas as pd

import scarf
    
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



def scarf_algo(preDATA_PATH,data_name,adata_input,class_nums,GTname):
    adata = adata_input.copy()

    # file_path = os.path.join(preDATA_PATH, data_name+'_adata_preprocess.h5ad')
    file_path = os.path.join(preDATA_PATH,data_name+'.h5ad')
    ## 读取h5ad数据
    reader = scarf.H5adReader(
        file_path, 
        cell_ids_key = 'index',               # Where Cell/barcode ids are saved under 'obs' slot
        feature_ids_key = 'index',            # Where gene ids are saved under 'var' slot
        feature_name_key = 'index'  # Where gene names are saved under 'var' slot
    )  

    print('reader.nCells, reader.nFeatures = ',reader.nCells, reader.nFeatures)

    # change value of `zarr_loc` to your choice of filename and path
    ## 把h5ad数据转化成zarr格式，才能后续计算

    writer = scarf.H5adToZarr(
        reader,
        zarr_loc=file_path+".zarr"
    )
    writer.dump()
    ds = scarf.DataStore(
        file_path+".zarr",
        nthreads=4,
        min_features_per_cell=10
    )

    ## 数据清洗和预处理
    ds.filter_cells(
        attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
        highs=[15000, 4000, 15],
        lows=[1000, 500, 0]
    )

    ## 开始聚类计算
    ds.RNA.normMethod = scarf.assay.norm_dummy
    ds.make_graph(feat_key='I', k=31, dims=25, n_centroids=100, show_elbow_plot=False)
    t0 = time.time()
    ds.run_clustering(n_clusters=class_nums)
    time_scarf=round(time.time()-t0,3)

    ### TYPE 2: to save clutering result
    clusts = pd.Series(ds.cells.fetch_all('RNA_cluster'))
    #最终聚类结果是"clusts"
    df_cluster_result = clusts.to_frame(name='scarf_cluster_result')
    ## 把聚类结果给到原始h5ad文件中。
    ###adata = sc.read(file_path)
    ## 获取adata的index，作为dataframe的一列.

    df1 = pd.DataFrame({'adata_index_name': adata.obs.index}) # 获得adata的index，变成一个dataframe
    df2 = pd.concat([df1, df_cluster_result], axis=1,join='outer') #按index数字顺序，拼接adata的index，以及scarf算法聚类结果
    df2 = df2.set_index('adata_index_name') #把adata的index变成这个dataframe的index， 方便整合到adata


    # adata.obs['scarf_labels_debug'] = df2['scarf_cluster_result']
    # adata.obs['scarf_labels_debug'] = adata.obs['scarf_labels_debug'].astype('category')


    # if data_name == 'paul15':
    #     adata.obs['paul15_label'] = adata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n=1, expand=True).values[:, 1]
    #     adata.obs['paul15_label'].astype('category')
    # if GTname is not None:
    #     import tools
    #     acc_results_value, acc_results_name = tools.evaluate(adata,'scarf_labels_debug',GTname)
    #     [print(see_info) for see_info in (acc_results_name,acc_results_value)]
    del df1
    del ds
    del adata
    return df2,time_scarf








#######################
DATA_PATH = r'D:\aaa\dataset\paper_DESC_dataset'
WORK_PATH = r'D:\CTEC_submit_work\review_BIOINF\project_algo_review\scarf_algo'
DATA_PATH  = win_to_linux(DATA_PATH)
WORK_PATH  = win_to_linux(WORK_PATH)



SAVE_PATH = WORK_PATH
scarf_name_pattern = '_processed_SCARF_res_'

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
save_path = os.path.join(SAVE_PATH,'result_scarf_'+str(EXP))
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
        adata.obs['paul15_label'] = adata.obs['paul15_label'].astype('category')
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
    adata_for_scarf = adata_pre.copy()
    # same PCA for all paras
    sc.tl.pca(adata_for_scarf, svd_solver='arpack',random_state=1) # use default paras
    sc.pp.neighbors(adata_for_scarf, n_neighbors=10, use_rep="X_pca", random_state=1) # use default paras
    print('adata_for_scarf:\n',adata_for_scarf)


    # [use scarf for result]
    print('    scarf method ......')
    ###adata_pre.write_h5ad(os.path.join(save_path, data_name+'_adata_preprocess.h5ad'))
    df_cluster_raw,time_scarf = scarf_algo(DATA_PATH,data_name,adata_for_scarf,class_nums,GTname)
    print("time_scarf  = ",time_scarf )
    # 去掉index包含nan的row
    df_cluster_raw['new_column'] = df_cluster_raw.index
    df_cluster = df_cluster_raw.dropna(subset=['new_column'])
    # 聚类标签从int改成str
    df_cluster['scarf_cluster_result'] = df_cluster['scarf_cluster_result'].astype(str)
    df_cluster.to_csv(os.path.join(save_path, data_name+'_cluster_scarf__df_cluster_time_'+str(time_scarf)+'_.csv'))



    # save cluster into adta
    ADATA_RAW.obs['scarf_labels'] = df_cluster['scarf_cluster_result']
    # del df_cluster
    ADATA_RAW.obs['scarf_labels'] = ADATA_RAW.obs['scarf_labels'].astype('category')
    res_scarf = 'input_class_nums'
    ADATA_RAW.obs.to_csv(os.path.join(save_path, data_name+scarf_name_pattern+'.csv'))
    ADATA_RAW.write_h5ad(os.path.join(save_path, data_name+'_cluster_scarf.h5ad'))


    # [evaluation]
    # try:
    if 1:
        ## [] calc acc
        acc_results_value, acc_results_name = evaluate(ADATA_RAW,'scarf_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_SCARF.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('Acc_Name,'+list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_scarf,'+list2str(acc_results_value)+','+str(res_scarf)+','+str(time_scarf)+'\n')
            f.write('\n')
        
        acc_algo = pd.read_csv(result_acc_path,delimiter = '\t')
        pd_dict.update({data_name:['method_name','ARI','NMI','resolution','time']})
        pd_dict.update({data_name+'_SCARF':acc_algo.iloc[-1,0].split(',')})
    # except:
    #     pass

# [evaluation]
acc_one_data = pd.DataFrame.from_dict(pd_dict,orient = 'index')
print('acc_one_data = \n',acc_one_data)
acc_one_data.to_csv(os.path.join(save_path,'result_acc====_Make_Acc_Table.csv'))

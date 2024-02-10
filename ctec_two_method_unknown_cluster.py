# -*- coding: utf-8 -*-

## import system file
from operator import is_not
import os
import scanpy as sc
import numpy as np
import time
import pandas as pd

## import algorithm file
from utils.algo_ctec import pre_process, cluster_leiden,find_resolution_leiden,find_resolution_louvain,evaluate,post_process_cluster_name_with_order_df
from utils.algo_ctec import algo_DESC
from utils.algo_ensemble_cv import cell_label_update_CV_df
from utils.algo_ensemble_od import cell_label_update_OD_df
from utils import calc_acc
from utils import path_util
from utils.tools import win_to_linux
import warnings
warnings.filterwarnings("ignore")
import gc
import argparse


from util_steps import *


#-------------------------------------------
# step 1: pre process
#-------------------------------------------
def step1(i,N_HIGH_VAR_TEST):
    DATASET_NUM = DATASET_NUM_LIST[i]
    print('\n\n','='*100)
    print('Step 1: pre_process')
    print('='*100)
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    t_start_dataset = time.time()


    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    N_HIGH_VAR = N_HIGH_VAR_TEST
    
    #[1.2 wash data]
    try:
        del adata,adata_desc,adata_leiden
    except:
        pass
    try:
        del adata_desc,adata_leiden
    except:
        pass

    adata = sc.read(os.path.join(DATA_PATH,data_name+data_ext)) 


    #[1.3 pre_process data]
    t0=time.time()
    print('    pre_process ......')
    # ADATA_RAW = adata.copy()
    # same pre-process for all paras
    Calc_Norm = True
    adata_pre = pre_process(adata, N_HIGH_VAR,BATCH_KEY,NORM=Calc_Norm)
    if DATASET_NUM==5:
        adata_pre.obs['paul15_label'] = adata_pre.obs['paul15_clusters'].str.split(
        "[0-9]{1,2}", n=1, expand=True).values[:, 1]
        adata_pre.obs['paul15_label'] = adata_pre.obs['paul15_label'].astype('category')
    
    print('adata_pre',adata_pre)
    


    #Step 1: save adata_pre
    
    PATH_adata_pre = os.path.join(save_path, data_name+'_adata_pre.h5ad')
    adata_pre.write_h5ad(PATH_adata_pre)
    t_end=time.time()-t0
    save_time_to_txt(PATH_adata_pre+'_TIME_step1.txt',t_end)

    if GTname is not None:
        ground_truth_df = adata_pre.obs[GTname].astype('category').to_frame()
        PATH_ground_truth_df = os.path.join(save_path, data_name+'_ground_truth_df.csv')
        ground_truth_df.to_csv(PATH_ground_truth_df) 

      
#-------------------------------------------
# step 2: calc PCA
#-------------------------------------------
def step2(i,n_neighbors):
    DATASET_NUM = DATASET_NUM_LIST[i]

    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, _,_,_,_,_,_,_ = loaddata_default(DATASET_NUM)

    print('\n\n','='*100)
    print('Step 2: calc PCA')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()
    
    #[1.2 wash data]

    #[1.3 pre_process data]
    print('    pre_process ......')
    PATH_adata_pre = os.path.join(save_path, data_name+'_adata_pre.h5ad')
    adata_pre = sc.read(PATH_adata_pre)
    print(adata_pre)

    #Step 2: save adata_for_leiden_desc; calc PCA etc.
    t0 = time.time()
    # same PCA for all paras
    sc.tl.pca(adata_pre, svd_solver='arpack',random_state=1) # use default paras
    # for DESC same neighbor, for Leiden, another neighbor
    sc.pp.neighbors(adata_pre, n_neighbors=n_neighbors, use_rep="X_pca", random_state=1) # use default paras
    
    
    PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
    # adata_pre.obsm['X_pca'] = adata_pre.obsm['X_pca'].astype(np.float16) #conver float32 to float16
    
    adata_pre.write_h5ad(PATH_adata_for_leiden_desc)
    adata_pre_obsm_X_pca = adata_pre.obsm['X_pca']
    np.save(os.path.join(save_path, data_name+'_adata_pre_obsm_X_pca.npy'),adata_pre_obsm_X_pca)
    print(adata_pre)
    t_end=time.time()-t0
    save_time_to_txt(PATH_adata_for_leiden_desc+'_TIME_step2.txt',t_end)


#-------------------------------------------
# step 3: Leiden algo
#-------------------------------------------
def step3(i):
    DATASET_NUM = DATASET_NUM_LIST[i]


    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, _,GTname,class_nums,_,_,res_leiden,_ = loaddata_default(DATASET_NUM)
    # res_leiden = None

    print('\n\n','='*100)
    print('Step 3: calc pure leiden')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()

    #Step 3: calc pure leiden
    if base_method_result_path is None:
        # re-calculate Leiden result
        PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
        adata_for_leiden = sc.read(PATH_adata_for_leiden_desc)
        if GTname is not None:
            df_GT = adata_for_leiden.obs[GTname].astype('category').to_frame()

        if res_leiden is None:
            res_leiden = find_resolution_leiden(adata_for_leiden, class_nums)

        print('\nres_leiden=',res_leiden)
        t0=time.time()
        sc.tl.leiden(adata_for_leiden,res_leiden)
        time_leiden=round(time.time()-t0,3)
        leiden_labels_df = adata_for_leiden.obs['leiden'].astype('category').to_frame()
        leiden_labels_df.rename(columns = {'leiden':'leiden_labels'}, inplace = True)
        del adata_for_leiden
    else:
        leiden_labels_df = pd.read_csv(os.path.join(base_method_result_path,data_name+'_leiden_labels_df.csv'),index_col=0)
        df_GT = pd.read_csv(os.path.join(base_method_result_path,data_name+'_ground_truth_df.csv'),index_col=0)
        time_leiden = 'na'


    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df.to_csv(PATH_leiden_labels_df) 
    save_time_to_txt(PATH_leiden_labels_df+'_TIME_step3.txt',time_leiden)
    



    ## [] calc acc
    if GTname is not None:
        DF_FOR_ACC = pd.concat([leiden_labels_df, df_GT], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC,'leiden_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_LEIDEN.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_LEIDEN,'+calc_acc.list2str(acc_results_value)+','+str(res_leiden)+','+str(time_leiden)+'\n')
            f.write('\n')



#-------------------------------------------
# step 4: DESC algo
#------------------------------------------
def step4(i,n_neighbors):
    DATASET_NUM = DATASET_NUM_LIST[i]
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    t_start_dataset = time.time()

    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, _,GTname,class_nums,_,_,_,res_louvain = loaddata_default(DATASET_NUM)
    # res_louvain = None

    print('\n\n','='*100)
    print('Step 4: calc pure DESC')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)

    ## Step 4: DESC
    if base_method_result_path is None:
        PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
        adata_for_desc = sc.read(PATH_adata_for_leiden_desc)
        if GTname is not None:
            # df_GT = adata_for_desc.obs[GTname].astype('category').to_frame()
            df_GT = adata_for_desc.obs[GTname].astype('category').to_frame()

        ## part 3: DESC result
        cpu_cores=16
        save_path_desc = os.path.join(save_path,data_name+'_desc')
        try:
            os.mkdir(save_path_desc)
        except:
            pass
        t0=time.time()
        

        if res_louvain is None:
            res_louvain, _ = find_resolution_louvain(adata_for_desc, class_nums,n_neighbors=n_neighbors)

        if adata_for_desc.n_obs <10000: #dataset cells nbr smaller than 10000 cells
            batch_size = 256
        else:
            batch_size = 1024
        print('batch_size = ',batch_size)

        desc_labels_df, res_desc = algo_DESC(adata_for_desc,FIND_RESO = False, res_lou=res_louvain, class_nums=class_nums,num_Cores=cpu_cores,save_path = save_path_desc,use_ae_weights=False, use_GPU=False,batch_size=batch_size)
        del adata_for_desc
        time_desc=round(time.time()-t0,3)
    else:
        desc_labels_df = pd.read_csv(os.path.join(base_method_result_path, data_name+'_desc_labels_df.csv'),index_col=0)
        df_GT = pd.read_csv(os.path.join(base_method_result_path,data_name+'_ground_truth_df.csv'),index_col=0)
        res_desc = res_louvain
        time_desc = 'na'

    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df.to_csv(PATH_desc_labels_df) 
    save_time_to_txt(PATH_desc_labels_df+'_TIME_step3.txt',time_desc)


    ## [] calc acc
    if GTname is not None:
        DF_FOR_ACC = pd.concat([desc_labels_df, df_GT], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC,'desc_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_DESC.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_DESC,'+calc_acc.list2str(acc_results_value)+','+str(res_desc)+','+str(time_desc)+'\n')
            f.write('\n')


#-------------------------------------------
# step 5: CTEC-DB
#-------------------------------------------
def step5(i):
    DATASET_NUM = DATASET_NUM_LIST[i]



    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, _,GTname,_,_,_,_,_ = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 5: calc pure CTEC_DB')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()



    ## step 5: CTEC-DB
    print('CTEC_DB')
    ## part 4: CTEC_DB
    algo_name = ['leiden_labels','desc_labels']
    algo_res  = [0,0]
    algo_time = [0,0]

    CV_THRESHOLD = 0.5
    ITER_NUMBER = 10
    t0=time.time()


    

    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 
    # leiden_labels_df = leiden_labels_df.set_index(keys = 'index')
    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)
    # desc_labels_df = desc_labels_df.set_index(keys = 'index') 


    ADATA_RAW_df = pd.concat([leiden_labels_df, desc_labels_df], axis=1, join="inner")

    ctec_db_labels_df = cell_label_update_CV_df(ADATA_RAW_df,algo_name,CV_THRESHOLD, ITER_NUMBER,print_info = False)
    time_ctec_db=round(time.time()-t0,3)
    print('<>'*10,'time_ctec_db = ',time_ctec_db)
    # ADATA_RAW.obs['ctec_db_labels'] = adata_ctec_db.obs['cluster_ensemble'].astype('category')
    # ADATA_RAW.obs['ctec_db_labels'] = adata_ctec_db_df['cluster_ensemble'].astype('category')
    # ADATA_RAW.obs.to_csv(os.path.join(save_path, data_name+'_processed_CTECDB.csv'))
    

    ctec_db_labels_df['cluster_ensemble'] =  pd.Categorical(ctec_db_labels_df['cluster_ensemble'])##.astype('category')
    ctec_db_labels_df.rename(columns = {'cluster_ensemble':'ctec_db_labels'}, inplace = True)
    ctec_db_labels_df['ctec_db_labels'] = post_process_cluster_name_with_order_df(ctec_db_labels_df,'ctec_db_labels', print_info=False)

    PATH_ctec_db_labels_df = os.path.join(save_path, data_name+'_ctec_db_labels_df.csv')
    ctec_db_labels_df.to_csv(PATH_ctec_db_labels_df) 

    try:
        if GTname is not None:
            PATH_ground_truth_df = os.path.join(save_path, data_name+'_ground_truth_df.csv')

            if base_method_result_path is None:
                ground_truth_df = pd.read_csv(PATH_ground_truth_df,index_col=0)
            else:
                ground_truth_df = pd.read_csv(os.path.join(base_method_result_path,data_name+'_ground_truth_df.csv'),index_col=0)

            # ground_truth_df = ground_truth_df.set_index(keys = 'index')
        ## [] calc acc
        DF_FOR_ACC1 = pd.concat([ctec_db_labels_df, ground_truth_df], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_db_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        
        print('\n---- Summary ----')
        DF_FOR_ACC2 = pd.concat([leiden_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo1, acc_results_name = evaluate_df(DF_FOR_ACC2,algo_name[0],GTname)
        print(acc_results_name,acc_algo1,"for algorithm Leiden")
        # [print(see_info) for see_info in (acc_results_name,acc_algo1)]
        
        DF_FOR_ACC3 = pd.concat([desc_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo2, acc_results_name = evaluate_df(DF_FOR_ACC3,algo_name[1],GTname)
        # [print(see_info) for see_info in (acc_results_name,acc_algo2)]
        print(acc_results_name,acc_algo2,"for algorithm DESC")

        acc_ctec_db, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_db_labels',GTname)
        # [print(see_info) for see_info in (acc_results_name,acc_ctec_db)]
        print(acc_results_name,acc_ctec_db,"for algorithm: CTEC-DB")
        
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_CTECDB.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('ITER_NUMBER,'+str(ITER_NUMBER)+'\n')
            f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_'+algo_name[0]+','+calc_acc.list2str(acc_algo1)+','+str(algo_res[0])+','+str(algo_time[0])+'\n')
            f.write('Acc_'+algo_name[1]+','+calc_acc.list2str(acc_algo2)+','+str(algo_res[1])+','+str(algo_time[1])+'\n')
            f.write('Acc_CTEC_DB,'+calc_acc.list2str(acc_ctec_db)+','+'NA ,'+str(time_ctec_db)+'\n')
            f.write('\n')
    except:
        print("No ground truth for this dataset")
        pass



#-------------------------------------------
# step 6: CTEC-OB
#-------------------------------------------
def step6(i):
    ## part 1: read and preprocesing
    #[1.1 read data]
    DATASET_NUM = DATASET_NUM_LIST[i]
    data_name, _,GTname,_,_,_,_,_ = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 6: calc pure CTEC_OB')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()


    ## step 6: CTEC-OB
    print('CTEC_OB:')
    algo_name = ['leiden_labels','desc_labels']
    algo_res  = [0,0]
    algo_time = [0,0]


    ITER_NUMBER = 20


    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 
    # leiden_labels_df = leiden_labels_df.set_index(keys = 'index')
    
    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)
    # desc_labels_df = desc_labels_df.set_index(keys = 'index') 
    
    ## load PCA
    if base_method_result_path is None:
        PATH_adata_PCA = os.path.join(save_path, data_name+'_adata_pre_obsm_X_pca.npy')
    else:
        PATH_adata_PCA = os.path.join(base_method_result_path, data_name+'_adata_pre_obsm_X_pca.npy')
    adata_pre_obsm_X_pca = np.load(PATH_adata_PCA)

    #the input dataframe
    ADATA_RAW_df = pd.concat([leiden_labels_df, desc_labels_df], axis=1, join="inner")

    #algorithm start
    t0=time.time()
    ctec_ob_labels_df = cell_label_update_OD_df(ADATA_RAW_df,adata_pre_obsm_X_pca,algo_name, ITER_NUMBER,print_info = False)
    time_ctec_ob=round(time.time()-t0,3)
    print('<>'*10,'time_ctec_ob = ',time_ctec_ob)

    
    ctec_ob_labels_df['cluster_ensemble'] =  pd.Categorical(ctec_ob_labels_df['cluster_ensemble'])##.astype('category')
    ctec_ob_labels_df.rename(columns = {'cluster_ensemble':'ctec_ob_labels'}, inplace = True)
    ctec_ob_labels_df['ctec_ob_labels'] = post_process_cluster_name_with_order_df(ctec_ob_labels_df,'ctec_ob_labels',print_info=False)

    PATH_ctec_ob_labels_df = os.path.join(save_path, data_name+'_ctec_ob_labels_df.csv')
    ctec_ob_labels_df.to_csv(PATH_ctec_ob_labels_df) 

    try:
        if GTname is not None:
            PATH_ground_truth_df = os.path.join(save_path, data_name+'_ground_truth_df.csv')
            if base_method_result_path is None:
                ground_truth_df = pd.read_csv(PATH_ground_truth_df,index_col=0)
                # ground_truth_df = ground_truth_df.set_index(keys = 'index')
            else:
                ground_truth_df = pd.read_csv(os.path.join(base_method_result_path,data_name+'_ground_truth_df.csv'),index_col=0)
                

        ## [] calc acc
        DF_FOR_ACC1 = pd.concat([ctec_ob_labels_df, ground_truth_df], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_ob_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        
        print('\n---- Summary ----')
        DF_FOR_ACC2 = pd.concat([leiden_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo1, acc_results_name = evaluate_df(DF_FOR_ACC2,algo_name[0],GTname)
        print(acc_results_name,acc_algo1,"for algorithm Leiden")

        DF_FOR_ACC3 = pd.concat([desc_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo2, acc_results_name = evaluate_df(DF_FOR_ACC3,algo_name[1],GTname)
        print(acc_results_name,acc_algo2,"for algorithm DESC")

        acc_ctec_ob, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_ob_labels',GTname)
        print(acc_results_name,acc_ctec_ob,"for algorithm: CTEC-OB")
        
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_CTECOB.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('ITER_NUMBER,'+str(ITER_NUMBER)+'\n')
            f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
            f.write('Acc_'+algo_name[0]+','+calc_acc.list2str(acc_algo1)+','+str(algo_res[0])+','+str(algo_time[0])+'\n')
            f.write('Acc_'+algo_name[1]+','+calc_acc.list2str(acc_algo2)+','+str(algo_res[1])+','+str(algo_time[1])+'\n')
            f.write('Acc_CTEC_OB,'+calc_acc.list2str(acc_ctec_ob)+','+'NA ,'+str(time_ctec_ob)+'\n')
            f.write('\n')
    except:
        print("No ground truth for this dataset")
        pass



#-------------------------------------------
# step 7: draw umap
#-------------------------------------------
def process_nan(adata,df0,name):
    adata.obs[name] = [np.nan]* adata.n_obs
    df1 = pd.DataFrame(adata.obs[name]) # np.nan
    df2 = pd.DataFrame(df0[name])
    df1[name] = df1[name].fillna(df2[name]) # np.nan
    # print(df1[name].isna().sum())
    adata.obs[name] =df1[name]
    adata_sub = adata[adata.obs[name]> -1]
    return adata_sub

def step7(i):
    DATASET_NUM = DATASET_NUM_LIST[i]
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from utils.umap_util import get_color_dict
    save_path_umap = os.path.join(save_path,'umap_result')
    try:
        os.mkdir(save_path_umap)
    except:
        pass

    FIG_DPI = 300
    t0=time.time()
    

    ## part 1: read and preprocesing
    #[1.1 read data]
    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 7: calc UMAP')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()
    ## load adata
    PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
    adata = sc.read(PATH_adata_for_leiden_desc)

    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 
    # leiden_labels_df = leiden_labels_df.set_index(keys = 'index')

    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)
    # desc_labels_df = desc_labels_df.set_index(keys = 'index') 

    # load dataframe CTEC-DB
    PATH_ctec_db_labels_df = os.path.join(save_path, data_name+'_ctec_db_labels_df.csv')
    ctec_db_labels_df = pd.read_csv(PATH_ctec_db_labels_df,index_col=0)
    # ctec_db_labels_df = ctec_db_labels_df.set_index(keys = 'index') 

    # load dataframe CTEC-OB
    PATH_ctec_ob_labels_df = os.path.join(save_path, data_name+'_ctec_ob_labels_df.csv')
    ctec_ob_labels_df = pd.read_csv(PATH_ctec_ob_labels_df,index_col=0) 
    # ctec_ob_labels_df = ctec_ob_labels_df.set_index(keys = 'index')




    plot_obsname = [GTname,'leiden_labels','desc_labels','ctec_db_labels','ctec_ob_labels']
    plot_showname = ['Reference standard annotation','Leiden','DESC','CTEC-DB','CTEC-OB']
    save_name = data_name
    SAVE_PDF = 1

    if int(adata.n_obs) != int(leiden_labels_df.shape[0]):
        adata = process_nan(adata,leiden_labels_df,'leiden_labels')

    adata.obs['leiden_labels'] = pd.Categorical(leiden_labels_df['leiden_labels'])
    adata.obs['desc_labels'] = pd.Categorical(desc_labels_df['desc_labels'])
    adata.obs['ctec_db_labels']  = pd.Categorical(ctec_db_labels_df['ctec_db_labels'])
    adata.obs['ctec_ob_labels']  = pd.Categorical(ctec_ob_labels_df['ctec_ob_labels'])


    sc.tl.umap(adata)
    sc.settings.set_figure_params(dpi_save=FIG_DPI, figsize=(8, 6), facecolor='white',)  # low dpi (dots per inch) yields small inline figures


    for i in range(0,len(plot_obsname)):
        try:
            adata.obs[plot_showname[i]] = adata.obs[plot_obsname[i]].astype(str)
        except:
            adata.obs[plot_showname[i]] = adata.obs[plot_obsname[i]].astype(str)

        # print('plot_cluster[i] = ',plot_obsname[i])
        # tmp1 = list(set(list(adata.obs[plot_obsname[i]])))
        # tmp2 = list(set(list(adata.obs[plot_showname[i]])))
        # try:
        #     tmp1.sort()
        #     tmp2.sort()
        # except:
        #     pass
        # # print('tmp1 = ',tmp1)
        # # print('tmp2 = ',tmp2)


    ## Plot UMAP with same color for different algo
    ## for Ground Truth result
    COLORLIST33 = ['#FFFF00','#1CE6FF','#FF34FF','#FF4A46','#008941','#006FA6','#A30059','#FFDBE5','#7A4900','#0000A6','#63FFAC','#B79762','#004D43','#8FB0FF','#997D87','#5A0007','#809693','#FEFFE6','#1B4400','#4FC601','#3B5DFF','#4A3B53','#FF2F80','#61615A','#BA0900','#6B7900','#00C2A0','#FFAA92','#FF90C9','#B903AA','#D16100','#DDEFFF','#000035']
    colorlist = COLORLIST33.copy()


    group_name = 'Reference standard annotation'
    gt_namelist =  list(set(adata.obs[group_name]))
    gt_namelist.sort()
    colordict_gt = dict()
    for i in range(0,len(gt_namelist)):
        colordict_gt[gt_namelist[i]] = colorlist[i]
    
    with plt.rc_context():  # Use this to set figure params like size and dpi
        sc.pl.umap(adata,
                color=group_name,
                palette=colordict_gt,
                legend_loc = 'on data',
                legend_fontsize = 'xx-small',
                legend_fontoutline=2)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_OnData.pdf'),bbox_inches='tight',dpi=FIG_DPI)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_OnData.png'),bbox_inches='tight',dpi=FIG_DPI)
    with plt.rc_context():  # Use this to set figure params like size and dpi        
        sc.pl.umap(adata,
                color=group_name,
                palette=colordict_gt,
                legend_loc = 'right margin',
                legend_fontsize = 'xx-small',
                legend_fontoutline=2)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_Right.pdf'),bbox_inches='tight',dpi=FIG_DPI)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_Right.png'),bbox_inches='tight',dpi=FIG_DPI)


    ## for different algorithm result
    for j in range(1,len(plot_showname)):
        group_name=plot_showname[j]
        print('\n','~'*30)
        print('group_name = ',group_name)
        name_gt = 'Reference standard annotation'
        ##################################################################
        colordict_gp = get_color_dict(adata,name_gt,group_name,colordict_gt,COLORLIST33.copy())
        ##################################################################

        with plt.rc_context():  # Use this to set figure params like size and dpi        
            sc.pl.umap(adata,
                    color=group_name,
                    palette=colordict_gp,
                    legend_loc = 'right margin',
                    legend_fontsize = 'xx-small',
                    legend_fontoutline=2)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_Right.pdf'),bbox_inches='tight',dpi=FIG_DPI)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_Right.png'),bbox_inches='tight',dpi=FIG_DPI)
        with plt.rc_context():  # Use this to set figure params like size and dpi        
            sc.pl.umap(adata,
                    color=group_name,
                    palette=colordict_gp,
                    legend_loc = 'on data',
                    legend_fontsize = 'xx-small',
                    legend_fontoutline=2)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_OnData.pdf'),bbox_inches='tight',dpi=FIG_DPI)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_2Algo_OnData.png'),bbox_inches='tight',dpi=FIG_DPI)
        # del colordict_gp
        print('umap image Saved:'+ os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_OnData......'))

    print('total calc time = ',time.time()-t0)









#### main code
# #-----------------------------------------------------
WORK_PATH = '/ctec_work'
DATA_PATH = os.path.join(WORK_PATH,'dataset')
# #-----------------------------------------------------

SAVE_PATH = WORK_PATH
leiden_name_pattern = '_LEIDEN_result_'
desc_name_pattern = '_DESC_result_'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id', type=int, default=0, required=False, help='ID of dataset')
parser.add_argument('--exp_id', type=str, default='demo_unknown_cluster', required=False, help='experiment id')
parser.add_argument('--base_method_result_path', type=str, default='./base_method_result/two_method_unknown_cluster', required=False, help='base_method_result_path')
args = parser.parse_args() 
EXP = args.exp_id
base_method_result_path = args.base_method_result_path




if args.dataset_id == 0:
    DATASET_NUM_LIST = [1,2,3,4,5]
else:
    DATASET_NUM_LIST = [args.dataset_id]
HVG_list = [1000]
N_NEIGHBORS = 10

for id in range(0,len(DATASET_NUM_LIST)):
    for HVG in HVG_list:
        ## manage result save path:
        save_path = os.path.join(SAVE_PATH,'Result_Ensemble_Two_Method_Exp_'+str(EXP))
        try:
            os.mkdir(save_path)
        except:
            pass
        print('~~~~~~~~~~     save_path = ',save_path)

        step1(id,HVG)
        step2(id,N_NEIGHBORS)
        step3(id)
        step4(id,N_NEIGHBORS)
        step5(id)
        step6(id)
        step7(id)
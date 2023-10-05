# -*- coding: utf-8 -*-
from operator import is_not
import os
import scanpy as sc
import numpy as np
import time
import pandas as pd


from utils.algo_ctec import pre_process,find_resolution_leiden,find_resolution_louvain,post_process_cluster_name_with_order_df
from utils.algo_ctec import algo_DESC,CTEC_DB_df,CTEC_OB_df
from utils import calc_acc
from utils.calc_acc import evaluate_df
from utils.path_util import win_to_linux, save_time_to_txt
import warnings
warnings.filterwarnings("ignore")
import argparse



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
        
    return data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain



#-------------------------------------------
# step 1: pre process
#-------------------------------------------
def step1(i):
    DATASET_NUM = DATASET_NUM_LIST[i]
    print('\n\n','='*100)
    print('Step 1: pre_process')
    print('='*100)
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    t_start_dataset = time.time()


    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)

    try:
        del adata,adata_desc,adata_leiden
    except:
        pass
    try:
        del adata_desc,adata_leiden
    except:
        pass

    adata = sc.read(os.path.join(DATA_PATH,data_name+data_ext))
    print(adata)

    t0=time.time()
    print('    pre_process ......')
    if DATASET_NUM==5:
        adata.obs['paul15_label'] = adata.obs['paul15_clusters'].str.split(
        "[0-9]{1,2}", n=1, expand=True).values[:, 1]
        adata.obs['paul15_label'] = adata.obs['paul15_label'].astype('category')
    if not GTname == None: 
            adata = adata[adata.obs[GTname].notna()] ###### find the nan from label, and remove them

    Calc_Norm = True
    adata_pre = pre_process(adata, N_HIGH_VAR,BATCH_KEY,NORM=Calc_Norm)
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
def step2(i):
    DATASET_NUM = DATASET_NUM_LIST[i]

    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 2: calc PCA')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()
    
    PATH_adata_pre = os.path.join(save_path, data_name+'_adata_pre.h5ad')
    adata_pre = sc.read(PATH_adata_pre)
    print(adata_pre)
    
    t0 = time.time()
    sc.tl.pca(adata_pre, svd_solver='arpack',random_state=1) # use default paras
    sc.pp.neighbors(adata_pre, n_neighbors=10, use_rep="X_pca", random_state=1) # use default paras
    
    PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
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

    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)

    print('\n\n','='*100)
    print('Step 3: calc pure leiden')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()


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
    leiden_labels_df = adata_for_leiden.obs['leiden'].astype("str").astype('category').to_frame()
    leiden_labels_df.rename(columns = {'leiden':'leiden_labels'}, inplace = True)
    del adata_for_leiden

    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df.to_csv(PATH_leiden_labels_df) 
    save_time_to_txt(PATH_leiden_labels_df+'_TIME_step3.txt',time_leiden)
    
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
#-------------------------------------------
def step4(i):
    DATASET_NUM = DATASET_NUM_LIST[i]
    t_start_dataset = time.time()

    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 4: calc pure DESC')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)

    PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
    adata_for_desc = sc.read(PATH_adata_for_leiden_desc)
    if GTname is not None:
        df_GT = adata_for_desc.obs[GTname].astype("str").astype('category').to_frame()

    cpu_cores=16
    save_path_desc = os.path.join(save_path,data_name+'_desc')
    try:
        os.mkdir(save_path_desc)
    except:
        pass
    t0=time.time()

    if res_louvain is None:
        res_louvain, _ = find_resolution_louvain(adata_for_desc, class_nums,n_neighbors=10)


    if adata_for_desc.n_obs <10000: #dataset cells nbr smaller than 10000 cells
        batch_size = 256
    else:
        batch_size = 1024
    print('batch_size = ',batch_size)
    desc_labels_df, res_desc = algo_DESC(adata_for_desc,FIND_RESO = False, res_lou=res_louvain, class_nums=class_nums,num_Cores=cpu_cores,save_path = save_path_desc,use_ae_weights=True,use_GPU=False,batch_size=batch_size)
    del adata_for_desc
    time_desc=round(time.time()-t0,3)

    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df.to_csv(PATH_desc_labels_df) 
    save_time_to_txt(PATH_desc_labels_df+'_TIME_step3.txt',time_desc)

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

    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 5: calc pure CTEC_DB')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()

    res_desc=res_louvain


    algo_name = ['leiden_labels','desc_labels']
    algo_res  = [res_leiden,res_desc]
    algo_time = [0,0]
    CV_THRESHOLD = 0.5
    ITER_NUMBER = 20
    t0=time.time()

    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 
    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)

    ADATA_RAW_df = pd.concat([leiden_labels_df, desc_labels_df], axis=1, join="inner")
    ctec_db_labels_df = CTEC_DB_df(ADATA_RAW_df,algo_name,CV_THRESHOLD, ITER_NUMBER)
    time_ctec_db=round(time.time()-t0,3)
    print('<>'*10,'time_ctec_db = ',time_ctec_db)
    

    ctec_db_labels_df['cluster_ensemble'] =  pd.Categorical(ctec_db_labels_df['cluster_ensemble'])
    ctec_db_labels_df.rename(columns = {'cluster_ensemble':'ctec_db_labels'}, inplace = True)
    ctec_db_labels_df['ctec_db_labels'] = post_process_cluster_name_with_order_df(ctec_db_labels_df,'ctec_db_labels')
    PATH_ctec_db_labels_df = os.path.join(save_path, data_name+'_ctec_db_labels_df.csv')
    ctec_db_labels_df.to_csv(PATH_ctec_db_labels_df) 

    try:
        if GTname is not None:
            PATH_ground_truth_df = os.path.join(save_path, data_name+'_ground_truth_df.csv')
            ground_truth_df = pd.read_csv(PATH_ground_truth_df,index_col=0) 

        DF_FOR_ACC1 = pd.concat([ctec_db_labels_df, ground_truth_df], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_db_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        
        print('\n---- Summary ----')
        DF_FOR_ACC2 = pd.concat([leiden_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo1, acc_results_name = evaluate_df(DF_FOR_ACC2,algo_name[0],GTname)
        print('___acc: ',algo_name[0])
        print(acc_results_name,acc_algo1)
        
        DF_FOR_ACC3 = pd.concat([desc_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo2, acc_results_name = evaluate_df(DF_FOR_ACC3,algo_name[1],GTname)
        print('___acc: ',algo_name[1])
        print(acc_results_name,acc_algo2)

        acc_ctec_db, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_db_labels',GTname)
        print('___acc: ctec_db_labels')
        print(acc_results_name,acc_ctec_db)
        
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
        pass


#-------------------------------------------
# step 6: CTEC-OB
#-------------------------------------------
def step6(i):
    DATASET_NUM = DATASET_NUM_LIST[i]
    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 6: calc pure CTEC_OB')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()

    res_desc=res_louvain
    algo_name = ['leiden_labels','desc_labels']
    algo_res  = [res_leiden,res_desc]
    algo_time = [0,0]

    ITER_NUMBER = 20

    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 
    
    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)
    
    ## load PCA
    PATH_adata_PCA = os.path.join(save_path, data_name+'_adata_pre_obsm_X_pca.npy')
    adata_pre_obsm_X_pca = np.load(PATH_adata_PCA)

    #the input dataframe
    ADATA_RAW_df = pd.concat([leiden_labels_df, desc_labels_df], axis=1, join="inner")

    #algorithm start
    t0=time.time()
    # ctec_ob_labels_df = cell_label_update_OD_df(ADATA_RAW_df,adata_pre_obsm_X_pca,algo_name, ITER_NUMBER)
    ctec_ob_labels_df = CTEC_OB_df(ADATA_RAW_df,adata_pre_obsm_X_pca,algo_name, ITER_NUMBER)

    time_ctec_ob=round(time.time()-t0,3)
    print('<>'*10,'time_ctec_ob = ',time_ctec_ob)

    
    ctec_ob_labels_df['cluster_ensemble'] =  pd.Categorical(ctec_ob_labels_df['cluster_ensemble'])
    ctec_ob_labels_df.rename(columns = {'cluster_ensemble':'ctec_ob_labels'}, inplace = True)
    ctec_ob_labels_df['ctec_ob_labels'] = post_process_cluster_name_with_order_df(ctec_ob_labels_df,'ctec_ob_labels')

    PATH_ctec_ob_labels_df = os.path.join(save_path, data_name+'_ctec_ob_labels_df.csv')
    ctec_ob_labels_df.to_csv(PATH_ctec_ob_labels_df) 

    try:
        if GTname is not None:
            PATH_ground_truth_df = os.path.join(save_path, data_name+'_ground_truth_df.csv')
            ground_truth_df = pd.read_csv(PATH_ground_truth_df,index_col=0) 

        DF_FOR_ACC1 = pd.concat([ctec_ob_labels_df, ground_truth_df], axis=1, join="inner")
        acc_results_value, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_ob_labels',GTname)
        [print(see_info) for see_info in (acc_results_name,acc_results_value)]
        
        print('\n---- Summary ----')
        DF_FOR_ACC2 = pd.concat([leiden_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo1, acc_results_name = evaluate_df(DF_FOR_ACC2,algo_name[0],GTname)
        print('___acc: ',algo_name[0])
        print(acc_results_name,acc_algo1)

        DF_FOR_ACC3 = pd.concat([desc_labels_df, ground_truth_df], axis=1, join="inner")
        acc_algo2, acc_results_name = evaluate_df(DF_FOR_ACC3,algo_name[1],GTname)
        print('___acc: ',algo_name[1])
        print(acc_results_name,acc_algo2)

        acc_ctec_ob, acc_results_name = evaluate_df(DF_FOR_ACC1,'ctec_ob_labels',GTname)
        print('___acc: ctec_ob_labels')
        print(acc_results_name,acc_ctec_ob)
        
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
        pass



#-------------------------------------------
# step 7: draw umap
#------------------------------------------
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

    data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 7: calc UMAP')
    print('DATASET_NUM=',DATASET_NUM,', EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()
    PATH_adata_for_leiden_desc = os.path.join(save_path, data_name+'_adata_for_leiden_desc.h5ad')
    adata = sc.read(PATH_adata_for_leiden_desc)

    ## load dataframe Leiden
    PATH_leiden_labels_df = os.path.join(save_path, data_name+'_leiden_labels_df.csv')
    leiden_labels_df = pd.read_csv(PATH_leiden_labels_df,index_col=0) 

    ## load dataframe DESC
    PATH_desc_labels_df = os.path.join(save_path, data_name+'_desc_labels_df.csv')
    desc_labels_df = pd.read_csv(PATH_desc_labels_df,index_col=0)

    ## load dataframe CTEC-DB
    PATH_ctec_db_labels_df = os.path.join(save_path, data_name+'_ctec_db_labels_df.csv')
    ctec_db_labels_df = pd.read_csv(PATH_ctec_db_labels_df,index_col=0)

    ## load dataframe CTEC-OB
    PATH_ctec_ob_labels_df = os.path.join(save_path, data_name+'_ctec_ob_labels_df.csv')
    ctec_ob_labels_df = pd.read_csv(PATH_ctec_ob_labels_df,index_col=0) 

    plot_obsname = [GTname,'leiden_labels','desc_labels','ctec_db_labels','ctec_ob_labels']
    plot_showname = ['Reference standard annotation','Leiden','DESC','CTEC-DB','CTEC-OB']
    save_name = data_name
    SAVE_PDF = 1

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
        # tmp1 = list(set(list(adata.obs[plot_obsname[i]])))
        # tmp2 = list(set(list(adata.obs[plot_showname[i]])))
        # try:
        #     tmp1.sort()
        #     tmp2.sort()
        # except:
        #     pass
        # print('tmp1 = ',tmp1)
        # print('tmp2 = ',tmp2)


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
        print('umap image Saved:'+ os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_OnData......'))

    print('total calc time = ',time.time()-t0)



def Make_Acc_Table(DATASET_NUM_LIST):
    # DATASET_NUM_LIST = [1,2,3,4,5]
    pd_dict = dict()
    for i in range(0,len(DATASET_NUM_LIST)):
        DATASET_NUM = DATASET_NUM_LIST[i]
        print('\n\n','='*100)
        print('DATASET_NUM=',DATASET_NUM)
        print('='*100)

        data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)

        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_LEIDEN.csv')
        acc_leiden = pd.read_csv(result_acc_path,delimiter = '\t')
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_DESC.csv')
        acc_desc = pd.read_csv(result_acc_path,delimiter = '\t')
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_CTECOB.csv')
        acc_ctecob = pd.read_csv(result_acc_path,delimiter = '\t')
        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_CTECDB.csv')
        acc_ctecdb = pd.read_csv(result_acc_path,delimiter = '\t')

        pd_dict.update({data_name:['method_name','ARI','NMI','resolution','time','data_name']})
        pd_dict.update({data_name+'_LEIDEN':acc_leiden.iloc[-1,0].split(',')})
        pd_dict.update({data_name+'_DESC':acc_desc.iloc[-1,0].split(',')})
        pd_dict.update({data_name+'_CTECOB':acc_ctecob.iloc[-1,0].split(',')})
        pd_dict.update({data_name+'_CTECDB':acc_ctecdb.iloc[-1,0].split(',')})

    acc_one_data = pd.DataFrame.from_dict(pd_dict,orient = 'index')
    print('acc_one_data = \n',acc_one_data)
    acc_one_data.to_csv(os.path.join(save_path,'result_acc====_Make_Acc_Table.csv'))
        



if __name__ == "__main__": 
    #-----------------------------------------------------
    # DATA_PATH = r'D:\dataset' 
    # WORK_PATH = r'D:\CTEC_algo' 
    # DATA_PATH  = win_to_linux(DATA_PATH)
    # WORK_PATH  = win_to_linux(WORK_PATH)
    # #-----------------------------------------------------
    WORK_PATH = './'
    DATA_PATH = os.path.join(WORK_PATH,'dataset')
    # #-----------------------------------------------------

    SAVE_PATH = WORK_PATH
    leiden_name_pattern = '_LEIDEN_result_'
    desc_name_pattern = '_DESC_result_'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, default=5, required=False, help='ID of dataset')
    parser.add_argument('--exp_id', type=str, default='0', required=False, help='experiment id')
    args = parser.parse_args()    
    EXP = args.exp_id
    DATASET_NUM_LIST = [args.dataset_id]
    ## manage result save path:
    save_path = os.path.join(SAVE_PATH,'Result_Ensemble_Two_Method_'+str(EXP))
    try:
        os.mkdir(save_path)
    except:
        pass
    print('~~~~~~~~~~ save_path = ',save_path)

    for id in range(0,len(DATASET_NUM_LIST)):
        step1(id)
        step2(id)
        step3(id)
        step4(id)
        step5(id)
        step6(id)
        step7(id)
        print('')
        

    # Make_Acc_Table(DATASET_NUM_LIST)


# -*- coding: utf-8 -*-

import os
import scanpy as sc
import numpy as np
import time
import pandas as pd
from utils import calc_acc


def pre_process(adata, N_HIGH_VAR,BATCH_KEY,NORM=True):
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

def pre_process_ver0(adata, N_HIGH_VAR,BATCH_KEY,NORM=True):
    t0=time.time()
    if NORM is True:
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

def cluster_leiden(adata_input,res):
    adata_ = adata_input.copy()
    sc.tl.pca(adata_, svd_solver='arpack') # use default paras
    sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X_pca") # use default paras
    t_start=time.time()
    sc.tl.leiden(adata_,res)
    try:
        print('Total Leiden time = ',time.time()-t_start)
    except:
        pass
    return adata_


def find_resolution_leiden(adata_input, n_clusters):
    t0=time.time()
    adata = adata_input.copy()
    del adata_input
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 100]

    sc.tl.pca(adata, svd_solver='arpack') # use default paras
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca") # use default paras

    while obtained_clusters != n_clusters and iteration < 100:
        current_res = sum(resolutions) / 2
        
        sc.tl.leiden(adata, resolution=current_res)
        labels = adata.obs['leiden']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res

        iteration = iteration + 1
        print(iteration, current_res, obtained_clusters)
    print('    find_resolution_leiden ......time:',time.time()-t0)
    return current_res



def find_resolution_louvain(adata_input, n_clusters,n_neighbors):
    t0=time.time()
    adata = adata_input.copy()
    del adata_input
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 100]

    sc.tl.pca(adata, svd_solver='arpack') # use default paras
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca") # use default paras

    while obtained_clusters != n_clusters and iteration < 100:
        current_res = sum(resolutions) / 2
        
        sc.tl.louvain(adata, resolution=current_res,random_state =0)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res

        iteration = iteration + 1
        
        print(iteration, current_res, obtained_clusters,"[iteration, current_res, obtained_clusters]")
    print('    find_resolution_louvain ......time:',time.time()-t0)
    return current_res, adata

def evaluate(adata,pred_name,GTname):
    df = adata.obs[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_acc.calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name


def algo_DESC(adata,
                FIND_RESO: bool = False,
                class_nums: int = None,
                tol: float = 0.005,
                n_neighbors:int = 10,
                batch_size:int = 256, 
                res_lou: float  = None,
                learning_rate:int = 150, # the parameter of tsne
                use_GPU: bool = False,
                num_Cores:int = 1, #for reproducible, only use 1 cpu
                num_Cores_tsne:int = 1,
                save_encoder_step:int =3,# save_encoder_weights is False, this parameter is not used
                save_path: str=None,
                use_ae_weights:  bool = False,
            ) -> None:
    
    import desc
    
    if FIND_RESO == True:
        res_lou, adata = find_resolution_louvain(adata, class_nums,n_neighbors)
    else:
        if res_lou == None:
            res_lou = 1.0

    t_start=time.time()
    

    if adata.X.shape[0] > 10000: # as the paper DESC defined
        DIM = 128
    else:
        DIM = 64
    print('    algo_DESC: layer DIM = ',DIM)
    adata=desc.train(adata,
                            dims=[adata.shape[1],DIM,32],
                            tol=tol,
                            n_neighbors=n_neighbors,
                            batch_size=batch_size, # better than 512 on large sample dataset like: PBMC, CORTEC, MACAQUE datasets
                            louvain_resolution=[res_lou],# not necessarily a list, you can only set one value, like, louvain_resolution=1.0
                            save_dir=save_path,
                            do_tsne=False,
                            learning_rate=learning_rate, # the parameter of tsne
                            use_GPU=use_GPU,
                            num_Cores=num_Cores, #for reproducible, only use 1 cpu
                            num_Cores_tsne=num_Cores_tsne,
                            save_encoder_weights=False,
                            save_encoder_step=save_encoder_step,# save_encoder_weights is False, this parameter is not used
                            use_ae_weights=use_ae_weights,
                            do_umap=False) #if do_uamp is False, it will don't compute umap coordiate

    try:
        print('Total DESC time = ',time.time()-t_start)
    except:
        pass

    pred_name='desc'+'_'+str(res_lou)
    #adata.obs['max.prob']=adata.uns["prob_matrix"+str(res_lou)].max(1)
    adata.obs['desc_labels'] = adata.obs[pred_name].astype("str").astype("category")
    
    #Computing maxmum probability
    # adata.write(os.path.join(save_path,"result_desc.h5ad"))

    ## save result
    #np.save(os.path.join(save_path,"feats.npy"), adata.X)
    #np.save(os.path.join(save_path,"embedding.npy"), adata.obsm['X_Embeded_z'+str(res_lou) ])
    # try:
    #     np.save(os.path.join(save_path,"probs.npy"), adata.uns["prob_matrix"+str(res_lou)].max(1))
    # except:
    #     pass
    # try:
    #     np.save(os.path.join(save_path,"probs.npy"), adata.uns.data['prob_matrix'+str(res_lou) ])
    # except:
    #     pass
    
    #adata.obs.to_csv(os.path.join(save_path,"infos.csv") )
    desc_labels_df = adata.obs['desc_labels'].astype('category').to_frame()
    return desc_labels_df,res_lou


def post_process_cluster_name_with_order(ann_data_ori, cluster_name,print_info = False):
    ann_data = ann_data_ori.copy()
    ann_data.obs[cluster_name] = pd.Categorical(ann_data.obs[cluster_name].astype(str))
    df_str_clstname_bad = ann_data.obs[cluster_name]
    df_int_clstname_good = df_str_clstname_bad.copy()
    clstname_unique = list(set(list(df_str_clstname_bad)))
    clstname_unique.sort()
    if print_info:
        print('clstname_unique = ',clstname_unique)


    dict_clstname_bad = dict()
    for i in range(0,len(clstname_unique)):
        dict_clstname_bad[clstname_unique[i]]=int(i)

    
    for key, value in dict_clstname_bad.items():
        print('key, value = ',key, value)
        df_int_clstname_good = df_int_clstname_good.replace(str(key), int(value))
        print(list(set(list(df_int_clstname_good))))
    
    
    df_str_clstname_good=df_int_clstname_good.apply(str)
    
    if print_info:
        print('clstname_bad  = ',list(set(list(df_str_clstname_bad))))
        print('clstname_good = ',list(set(list(df_str_clstname_good))))
    
    ann_data.obs[cluster_name] = pd.Categorical(df_str_clstname_good)
    del ann_data_ori
    return ann_data


def post_process_cluster_name_with_order_df(ann_data_ori_df, cluster_name,print_info = False):
    df_str_clstname_bad = ann_data_ori_df[cluster_name].astype(str)
    df_int_clstname_good = df_str_clstname_bad.copy()
    clstname_unique = list(set(list(df_str_clstname_bad)))
    clstname_unique.sort()
    if print_info:
        print('clstname_unique = ',clstname_unique)

    dict_clstname_bad = dict()
    for i in range(0,len(clstname_unique)):
        dict_clstname_bad[clstname_unique[i]]=int(i)

    for key, value in dict_clstname_bad.items():
        # print('key, value = ',key, value)
        df_int_clstname_good = df_int_clstname_good.replace(str(key), int(value))
        # print('df_int_clstname_good = ',list(set(list(df_int_clstname_good))))

    df_str_clstname_good=df_int_clstname_good.apply(str)
    return df_str_clstname_good

    
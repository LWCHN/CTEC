# -*- coding: utf-8 -*-

import os
import scanpy as sc
import numpy as np
import time
import pandas as pd
from utils import calc_acc
from scipy.optimize import linear_sum_assignment
from pyod.models.copod import COPOD #for CTEC_OB


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
                random_seed = 20231005,
            ) -> None:
    
    import scanpy as sc
    import desc
    import time
    


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
                            do_umap=False,
                            random_seed = random_seed) #if do_uamp is False, it will don't compute umap coordiate

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

def post_process_cluster_name_with_order(ann_data_ori, cluster_name):
    ann_data = ann_data_ori.copy()
    ann_data.obs[cluster_name] = pd.Categorical(ann_data.obs[cluster_name].astype(str))
    df_str_clstname_bad = ann_data.obs[cluster_name]
    df_int_clstname_good = df_str_clstname_bad.copy()
    clstname_unique = list(set(list(df_str_clstname_bad)))
    clstname_unique.sort()

    dict_clstname_bad = dict()
    for i in range(0,len(clstname_unique)):
        dict_clstname_bad[clstname_unique[i]]=int(i)

    for key, value in dict_clstname_bad.items():
        # print('key, value = ',key, value)
        df_int_clstname_good = df_int_clstname_good.replace(str(key), int(value))
        # print(list(set(list(df_int_clstname_good))))

    df_str_clstname_good=df_int_clstname_good.apply(str)

    ann_data.obs[cluster_name] = pd.Categorical(df_str_clstname_good)
    del ann_data_ori
    return ann_data


def post_process_cluster_name_with_order_df(ann_data_ori_df, cluster_name):
    df_str_clstname_bad = ann_data_ori_df[cluster_name].astype(str)
    df_int_clstname_good = df_str_clstname_bad.copy()
    clstname_unique = list(set(list(df_str_clstname_bad)))
    clstname_unique.sort()

    dict_clstname_bad = dict()
    for i in range(0,len(clstname_unique)):
        dict_clstname_bad[clstname_unique[i]]=int(i)

    for key, value in dict_clstname_bad.items():
        df_int_clstname_good = df_int_clstname_good.replace(str(key), int(value))
    
    df_str_clstname_good=df_int_clstname_good.apply(str)
    return df_str_clstname_good

    

#------------------------------------------------------------------
#------------------------------------------------------------------
### Algo ensemble by coefficient of variation 
def CTEC_DB_df(adata_df_input,algo_name,cv_threshold,ITER_NUM=0,SHOW_CTAB=False):
    adata_df_output = adata_df_input.copy()
    adata_df_calc = adata_df_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    print('--------------  ITER_NUM',ITER_NUM)
    
    for iter in range(0,ITER_NUM+1):
        if iter==0:
            adata_df_calc.loc[:,['algo1']] = pd.Categorical(adata_df_input[algo_name[0]]) # init C_ref
            df_C_ref = pd.DataFrame(adata_df_input[algo_name[0]])
            adata_df_calc.loc[:,['algo2']] = pd.Categorical(adata_df_input[algo_name[1]]) # init C_asst
        else: # use new result to assit main algo(algo1)
            df_C_ref = pd.DataFrame(adata_df_correct_C_ref['cluster_ensemble'])

        ## Show cross table each iter
        if SHOW_CTAB:
            print(pd.crosstab(adata_df_calc['algo1'],adata_df_calc['algo2']))

        ## input for c_tab
        adata_df_correct_C_ref = correct_ann_db_df(adata_df_calc, 'algo1','algo2',cv_threshold)
        adata_df_correct_C_asst = correct_ann_db_df(adata_df_calc, 'algo2','algo1',cv_threshold)

        ## show ARI, NMI if possible
        adata_df_output.loc[:,['cluster_ensemble']] = adata_df_correct_C_ref['cluster_ensemble']
        # if one_gt_name!=None:
        #     DF_FOR_ACC = adata_df_output[[ 'cluster_ensemble',one_gt_name]]        

        # update result for next iteration
        adata_df_calc.loc[:,['algo1']] = adata_df_correct_C_ref['cluster_ensemble'] # update C'ref
        adata_df_calc.loc[:,['algo2']] = adata_df_correct_C_asst['cluster_ensemble'] # update C'asst

        df_C_prime_ref = pd.DataFrame(adata_df_correct_C_ref['cluster_ensemble'])
        aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
        bb = np.array(df_C_ref.iloc[:,0].apply(str))
        case_same = np.where(aa==bb)[0].shape[0]
        case_total = df_C_prime_ref.shape[0]
        
        # if one_gt_name!=None:
        #     print('iter = ',iter, calc_acc.calc_all_acc_simple(DF_FOR_ACC,gt_name=one_gt_name,pred_name='cluster_ensemble',decimals=3),'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
        # else:
        #     print('iter = ',iter,'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))

        ratio_before_after.append(1.0*case_same/case_total)
        if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
            print('iter = ',iter)
            break
    # try:
    #     print('Total CTEC-DB time = ',time.time()-t_start)
    # except:
    #     pass
    
    return adata_df_output

def correct_ann_db_df(anndata_df, method_1_name, method_2_name, cv_threshold):
    merge_df = anndata_df.copy()
    pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
    ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
    df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)

    if 'cluster_ensemble' in anndata_df:
        anndata_df = anndata_df.drop(columns="cluster_ensemble")

    anndata_df.loc[:,['cluster_ensemble']]  = df_new_pred['cluster_ensemble']
    return anndata_df

def get_correct_name(ctab, cv_threshold):
    cv_column = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=0).values  #apply function to each column.
    cv_row = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=1).values

    ctab_column_correct_name = ctab.columns.values[cv_column >= cv_threshold] # default method cv > 2.0
    ctab_row_correct_name = ctab.index.values[cv_row >= cv_threshold] # default method cv > 2.0
    return(ctab_row_correct_name, ctab_column_correct_name)

def correct_ctab(merge_table, ctab, ctab_column_correct_name):
    max_pos_row = ctab[ctab_column_correct_name].apply(np.argmax, axis=0).values
    max_pos_row_name_index = ctab.index.values[max_pos_row]
    
    tmp_table_mod_row = merge_table.copy()
    if 'cluster_ensemble' in tmp_table_mod_row:
        tmp_table_mod_row = tmp_table_mod_row.drop(columns="cluster_ensemble")

    for i,j in zip(max_pos_row_name_index, ctab_column_correct_name):
        index = tmp_table_mod_row[ctab.columns.name] == j  # all the sample belongs to some single desc cluster
        tmp_table_mod_row.loc[index, ctab.index.name] = i
    
    tmp_table_mod_row = tmp_table_mod_row.rename(columns={ctab.index.name:'cluster_ensemble'})
    return tmp_table_mod_row




# def correct_ann_db(anndata, method_1_name, method_2_name, cv_threshold):
#     adata_input = anndata.copy()
#     merge_df = adata_input.obs[[ method_1_name, method_2_name ]]  
#     pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
#     ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
#     df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)
#     adata_input.obs['cluster_ensemble'] = df_new_pred['cluster_ensemble']
#     return adata_input

# def CTEC_DB(adata_input,algo_name,one_gt_name,cv_threshold,ITER_NUM=0):
#     adata_output = adata_input.copy()
#     adata_calc = adata_input.copy()
#     t_start=time.time()
#     ratio_before_after = list()
#     ITER_NUM=ITER_NUM
#     print('--------------  ITER_NUM',ITER_NUM)
    
#     for iter in range(0,ITER_NUM+1):
#         if iter==0:
#             adata_calc.obs['algo1'] = pd.Categorical(adata_input.obs[algo_name[0]]) # init C_ref
#             df_C_ref = pd.DataFrame(adata_input.obs[algo_name[0]])
#             adata_calc.obs['algo2'] = pd.Categorical(adata_input.obs[algo_name[1]]) # init C_asst
#         else: # use new result to assit main algo(algo1)
#             df_C_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])

#         ## Show cross table each iter
#         print(pd.crosstab(adata_calc.obs['algo1'],adata_calc.obs['algo2']))

#         ## input for c_tab
#         adata_correct_C_ref = correct_ann_db(adata_calc, 'algo1','algo2',cv_threshold)
#         adata_correct_C_asst = correct_ann_db(adata_calc, 'algo2','algo1',cv_threshold)

#         ## show ARI, NMI if possible
#         adata_output.obs['cluster_ensemble'] = adata_correct_C_ref.obs['cluster_ensemble']
#         if one_gt_name!=None:
#             df = adata_output.obs[[ 'cluster_ensemble',one_gt_name]]        
#             # print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble'))

#         # update result for next iteration
#         adata_calc.obs['algo1'] = adata_correct_C_ref.obs['cluster_ensemble'] # update C'ref
#         adata_calc.obs['algo2'] = adata_correct_C_asst.obs['cluster_ensemble'] # update C'asst

#         df_C_prime_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])
#         aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
#         bb = np.array(df_C_ref.iloc[:,0].apply(str))
#         case_same = np.where(aa==bb)[0].shape[0]
#         case_total = df_C_prime_ref.shape[0]
        
#         if one_gt_name!=None:
#             print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble',decimals=3),'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
#         else:
#             print('iter = ',iter,'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))

#         ratio_before_after.append(1.0*case_same/case_total)
#         if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
#             print('iter = ',iter)
#             break

#     try:
#         print('Total CTEC-DB time = ',time.time()-t_start)
#     except:
#         pass

#     return adata_output



# def merge_table(df_1, df_1_name, df_1_method, df_2, df_2_name, df_2_method):
#     df1_tmp = pd.DataFrame()
#     df1_tmp[df_1_method] = df_1[df_1_method].astype(str)
#     df1_tmp.set_index(df_1[df_1_name].astype(str), inplace= True)

#     df2_tmp = pd.DataFrame()
#     df2_tmp[df_2_method] = df_2[df_2_method].astype(str)
#     df2_tmp.set_index(df_2[df_2_name].astype(str), inplace= True)

#     df1_tmp.index.name = "cellname"
#     df2_tmp.index.name = "cellname"

#     merge_df = df1_tmp.join(df2_tmp)
#     return(merge_df)




#------------------------------------------------------------------
#------------------------------------------------------------------
### Algo ensemble by outlier detection
def CTEC_OB_df(adata_df_input,adata_PCA,algo_name,ITER_NUM=0,SHOW_CTAB=False):
    adata_df_output = adata_df_input.copy()
    adata_df_calc = adata_df_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    print('--------------  ITER_NUM total= ',ITER_NUM)
    
    for iter in range(0,ITER_NUM+1):
        if iter==0:
            adata_df_calc.loc[:,['algo1']] = pd.Categorical(adata_df_input[algo_name[0]]) # init C_ref
            df_C_ref = pd.DataFrame(adata_df_input[algo_name[0]])
            adata_df_calc.loc[:,['algo2']] = pd.Categorical(adata_df_input[algo_name[1]]) # init C_asst
            del adata_df_input
        else: # use new result to assit main algo(algo1)
            df_C_ref = pd.DataFrame(adata_df_correct_C_ref['cluster_ensemble'])
        
        ## Show cross table each iter
        if SHOW_CTAB:
            print(pd.crosstab(adata_df_calc['algo1'],adata_df_calc['algo2']))

        ## input for c_tab
        adata_df_correct_C_ref = correct_ann_by_pca_df(adata_df_calc,adata_PCA, 'algo1','algo2')
        adata_df_correct_C_asst = correct_ann_by_pca_df(adata_df_calc,adata_PCA, 'algo2','algo1')

        ## show ARI, NMI if possible
        adata_df_output.loc[:,['cluster_ensemble']] = adata_df_correct_C_ref['cluster_ensemble']

        # update result for next iteration
        adata_df_calc.loc[:,['algo1']]= adata_df_correct_C_ref['cluster_ensemble'] # update C'ref
        adata_df_calc.loc[:,['algo2']] = adata_df_correct_C_asst['cluster_ensemble'] # update C'asst

        df_C_prime_ref = pd.DataFrame(adata_df_correct_C_ref['cluster_ensemble'])
        aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
        bb = np.array(df_C_ref.iloc[:,0].apply(str))
        case_same = np.where(aa==bb)[0].shape[0]
        case_total = df_C_prime_ref.shape[0]
        if SHOW_CTAB:
            print('iter = ',iter,'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
        
        ratio_before_after.append(1.0*case_same/case_total)  
        if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
            print('iter = ',iter)
            break

    # try:
    #     print('Total CTEC-OB time = ',time.time()-t_start)
    # except:
    #     pass
    return adata_df_output



def correct_ann_by_pca_df(anndata_df, adata_PCA, method_1_name, method_2_name):
    new_anndata_df = anndata_df.copy()
    del anndata_df
    # new_anndata_df[method_1_name] = new_anndata_df[method_1_name].astype(str).astype("category")
    # new_anndata_df[method_2_name] = new_anndata_df[method_2_name].astype(str).astype("category")

    new_anndata_df_cross = pd.crosstab(new_anndata_df[method_1_name],new_anndata_df[method_2_name])

    max_pos_row = new_anndata_df_cross.apply(np.argmax, axis=0).values
    assert len(max_pos_row) == new_anndata_df_cross.shape[1], "please check you max position"

    for i in range(len(max_pos_row)):
        train_index = np.logical_and(new_anndata_df[method_2_name] == new_anndata_df_cross.columns[i],
                                     new_anndata_df[method_1_name] == new_anndata_df_cross.index.values[max_pos_row[i]])
        
        test_index = np.logical_and(new_anndata_df[method_2_name] == new_anndata_df_cross.columns[i],
                                    new_anndata_df[method_1_name] != new_anndata_df_cross.index.values[max_pos_row[i]])
        
        tmp_train = adata_PCA[train_index]
        tmp_test = adata_PCA[test_index]

        if tmp_test.shape[0] == 0:
            continue
        else:
            clf = COPOD()
            clf.fit(tmp_train)
            #y_train_scores = clf.decision_scores_ 
            #y_test_scores = clf.decision_function(tmp_test)
            y_test_pred = clf.predict(tmp_test)

            new_anndata_df.loc[new_anndata_df.index.isin(test_index.index[test_index][y_test_pred==0]), method_1_name] = new_anndata_df_cross.index.values[max_pos_row[i]]
    
    new_anndata_df['cluster_ensemble'] = new_anndata_df[method_1_name]
    return new_anndata_df



########################################################################################
### Algo ensemble by coefficient of variation ##########################################
########################################################################################
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score

# def correct_ctab(merge_table, ctab, ctab_column_correct_name):
#     max_pos_row = ctab[ctab_column_correct_name].apply(np.argmax, axis=0).values
#     max_pos_row_name_index = ctab.index.values[max_pos_row]
    
#     tmp_table_mod_row = merge_table.copy()

#     for i,j in zip(max_pos_row_name_index, ctab_column_correct_name):
#         index = tmp_table_mod_row[ctab.columns.name] == j  # all the sample belongs to some single desc cluster
#         tmp_table_mod_row.loc[index, ctab.index.name] = i
    
#     tmp_table_mod_row = tmp_table_mod_row.rename(columns={ctab.index.name:'cluster_ensemble'}) 
#     return tmp_table_mod_row

# def get_correct_name(ctab, cv_threshold):
#     cv_column = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=0).values  #apply function to each column.
#     cv_row = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=1).values

#     ctab_column_correct_name = ctab.columns.values[cv_column >= cv_threshold] # default method cv > 2.0
#     ctab_row_correct_name = ctab.index.values[cv_row >= cv_threshold] # default method cv > 2.0
#     return(ctab_row_correct_name, ctab_column_correct_name)


## replace by "def correct_ann_db_df()"
# def correct_ann_cv(df, method_1_name, method_2_name, cv_threshold):
#     merge_df = df[[ method_1_name, method_2_name ]]  
#     pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
#     ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
#     df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)
#     df['cluster_ensemble'] = df_new_pred['cluster_ensemble']
#     return df

## "def cell_label_update_CV_multiple" to "CTEC_DB_multiple"
def CTEC_DB_multiple(adata_input,algo_name,cv_threshold,ITER_NUM=0,PRINT_INFO=True):
    adata_output = adata_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    if PRINT_INFO: print('--------------  ITER_NUM',ITER_NUM)

    cluster_list = [adata_output.obs[algo_name[i]] for i in range(len(algo_name))]
    n =  len(cluster_list) # number of clustering results to ensemble
    use_ari = True
    while n > 1:
        if use_ari:
            # calculate pairwise ARI
            ARI_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    ARI_list.append((i, j, adjusted_rand_score(cluster_list[i], cluster_list[j])))
            #ARI_list.sort(key=lambda x: x[2], reverse=True)
            ARI_list.sort(key=lambda x: x[2], reverse=False) # ensemble small ARI first

            if PRINT_INFO: print('original algo_name:',algo_name)
            if PRINT_INFO: print('cluster_list_name = ',[one_series.name for one_series in cluster_list])
            if PRINT_INFO: print('n=', n, ' ARI_list=', ARI_list)
            # ensemble the two results with the smallest ARI
            #id_a = ARI_list[0][1] # new ensemble results always in the end of the list, used as ref 
            #id_b = ARI_list[0][0]
            id_a = ARI_list[0][0]
            id_b = ARI_list[0][1]

        else: # select the two results with the lowest CH score
            score_list = []
            for i in range(n):
                ch_score = calinski_harabasz_score(adata_input.obsm['X_pca'], cluster_list[i])
                score_list.append((i, ch_score))
            score_list.sort(key=lambda x: x[1], reverse=True)
            if PRINT_INFO: print('score_list = ',score_list)
            id_a = score_list[0][0]
            id_b = score_list[1][0]

        res_a = cluster_list[id_a]
        res_b = cluster_list[id_b]
        del cluster_list[max(id_a, id_b)] #the two method tobe ensemble in the following, so del them, remove the max firstly, then the index of min will no change in new list.
        del cluster_list[min(id_a, id_b)] #the two method tobe ensemble in the following, so del them first

        ## save the two method Pred cluster, they will be ensembled in the following, with new name 'cluster_ensemble'.
        ## and that is why the two method Pred cluster are del in 'cluster_list'
        temp = pd.DataFrame({'algo1':res_a, 'algo2':res_b})
        for iter in range(0,ITER_NUM+1):
            if iter==0:
                df_C_ref = pd.DataFrame(res_a)
            else: # use new result to assit main algo(algo1)
                df_C_ref = pd.DataFrame(correct_C_ref['cluster_ensemble'])

            ## input for c_tab
            correct_C_ref = correct_ann_db_df(temp, 'algo1','algo2',cv_threshold)
            correct_C_asst = correct_ann_db_df(temp, 'algo2','algo1',cv_threshold)

            # update result for next iteration
            temp['algo1'] = correct_C_ref['cluster_ensemble'] # update C'ref
            temp['algo2'] = correct_C_asst['cluster_ensemble'] # update C'asst

            df_C_prime_ref = pd.DataFrame(correct_C_ref['cluster_ensemble'])
            aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
            bb = np.array(df_C_ref.iloc[:,0].apply(str))
            case_same = np.where(aa==bb)[0].shape[0]
            case_total = df_C_prime_ref.shape[0]
       
            if PRINT_INFO:
                print('iter = ',iter, '; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
            ratio_before_after.append(1.0*case_same/case_total)
            if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
                if PRINT_INFO: print('early stop at iter = ',iter)
                break
        cluster_list.append(correct_C_ref['cluster_ensemble'])
        n =  len(cluster_list) 

    try:
        if PRINT_INFO: print('Total CTEC_DB time = ',time.time()-t_start)
    except:
        pass
    adata_output.obs['cluster_ensemble'] = cluster_list[0]
    return adata_output


def merge_table(df_1, df_1_name, df_1_method, df_2, df_2_name, df_2_method):
    df1_tmp = pd.DataFrame()
    df1_tmp[df_1_method] = df_1[df_1_method].astype(str)
    df1_tmp.set_index(df_1[df_1_name].astype(str), inplace= True)

    df2_tmp = pd.DataFrame()
    df2_tmp[df_2_method] = df_2[df_2_method].astype(str)
    df2_tmp.set_index(df_2[df_2_name].astype(str), inplace= True)

    df1_tmp.index.name = "cellname"
    df2_tmp.index.name = "cellname"

    merge_df = df1_tmp.join(df2_tmp)

    return(merge_df)



##############################################################################
### Algo ensemble by outlier detection #######################################
##############################################################################
def CTEC_OB_multiple(adata_input,algo_name,ITER_NUM=0, PRINT_INFO=True):
    adata_output = adata_input.copy()
    adata_calc = adata_input.copy()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    if PRINT_INFO: print('--------------  ITER_NUM total= ',ITER_NUM)
    t_start=time.time()
    
    cluster_list = [adata_output.obs[algo_name[i]] for i in range(len(algo_name))]
    n =  len(cluster_list) # number of clustering results to ensemble
    use_ari = True
    while n > 1:
        if use_ari:
            # calculate pairwise ARI
            ARI_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    ARI_list.append((i, j, adjusted_rand_score(cluster_list[i], cluster_list[j])))
            #ARI_list.sort(key=lambda x: x[2], reverse=True)
            ARI_list.sort(key=lambda x: x[2], reverse=False) # ensemble small ARI first

            if PRINT_INFO: print('original algo_name:',algo_name)
            if PRINT_INFO: print('cluster_list_name = ',[one_series.name for one_series in cluster_list])
            if PRINT_INFO: print('n=', n, ' ARI_list=', ARI_list)
            # ensemble the two results with the smallest ARI
            id_a = ARI_list[0][0]
            id_b = ARI_list[0][1]

        else: # select the two results with the lowest CH score
            score_list = []
            for i in range(n):
                ch_score = calinski_harabasz_score(adata_input.obsm['X_pca'], cluster_list[i])
                score_list.append((i, ch_score))
            score_list.sort(key=lambda x: x[1], reverse=True)
            id_a = score_list[0][0]
            id_b = score_list[1][0]

        res_a = cluster_list[id_a]
        res_b = cluster_list[id_b]
        del cluster_list[max(id_a, id_b)]
        del cluster_list[min(id_a, id_b)]

        for iter in range(0,ITER_NUM+1):
            if iter==0:
                adata_calc.obs['algo1'] = res_a # init C_ref
                df_C_ref = pd.DataFrame(res_a)
                adata_calc.obs['algo2'] = res_b # init C_asst
            else: # use new result to assit main algo(algo1)
                df_C_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])

            ## input for c_tab
            adata_correct_C_ref = correct_ann_by_pca(adata_calc, 'algo1','algo2')
            adata_correct_C_asst = correct_ann_by_pca(adata_calc, 'algo2','algo1')

            ## show ARI, NMI if possible
            adata_output.obs['cluster_ensemble'] = adata_correct_C_ref.obs['cluster_ensemble']

            # update result for next iteration
            adata_calc.obs['algo1'] = adata_correct_C_ref.obs['cluster_ensemble'] # update C'ref
            adata_calc.obs['algo2'] = adata_correct_C_asst.obs['cluster_ensemble'] # update C'asst

            df_C_prime_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])
            aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
            bb = np.array(df_C_ref.iloc[:,0].apply(str))
            case_same = np.where(aa==bb)[0].shape[0]
            case_total = df_C_prime_ref.shape[0]
            ratio_before_after.append(1.0*case_same/case_total)

            if PRINT_INFO:
                print('iter = ',iter, '; ratio_before_after = ', np.round(ratio_before_after[-1], decimals=5))  
            # print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble'), 
            #         '; ratio_before_after = ', np.round(ratio_before_after[-1], decimals=5))

            if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
                if PRINT_INFO: print('early stop at iter = ',iter)
                break
            # if 1.0*case_same/case_total >= 0.999:
        #     print('iter = ',iter)
        #     break

        cluster_list.append(adata_correct_C_ref.obs['cluster_ensemble'])
        n =  len(cluster_list) 
    
    try:
        if PRINT_INFO: print('Total CTEC_OB time = ',time.time()-t_start)
    except:
        pass
    return adata_output

def correct_ann_by_pca(anndata, method_1_name, method_2_name):
    new_anndata = anndata.copy()
    new_anndata.obs[method_1_name] = new_anndata.obs[method_1_name].astype(str).astype("category")
    new_anndata.obs[method_2_name] = new_anndata.obs[method_2_name].astype(str).astype("category")

    new_anndata_cross = pd.crosstab(new_anndata.obs[method_1_name],new_anndata.obs[method_2_name])

    max_pos_row = new_anndata_cross.apply(np.argmax, axis=0).values
    assert len(max_pos_row) == new_anndata_cross.shape[1], "please check you max position"

    for i in range(len(max_pos_row)):
        train_index = np.logical_and(new_anndata.obs[method_2_name] == new_anndata_cross.columns[i],
                                     new_anndata.obs[method_1_name] == new_anndata_cross.index.values[max_pos_row[i]])
        tmp_train = new_anndata[train_index].obsm["X_pca"] # ！！！mod
        #tmp_train = new_anndata[train_index].X

        test_index = np.logical_and(new_anndata.obs[method_2_name] == new_anndata_cross.columns[i],
                                    new_anndata.obs[method_1_name] != new_anndata_cross.index.values[max_pos_row[i]])
        tmp_test = new_anndata[test_index].obsm["X_pca"] # ！！！mod
        #tmp_test = new_anndata[test_index].X

        if tmp_test.shape[0] == 0:
            continue
        else:
            clf = COPOD()
            clf.fit(tmp_train)
            #y_train_scores = clf.decision_scores_ 
            #y_test_scores = clf.decision_function(tmp_test)
            y_test_pred = clf.predict(tmp_test)

            new_anndata.obs.loc[new_anndata.obs.index.isin(test_index.index[test_index][y_test_pred==0]), method_1_name] = new_anndata_cross.index.values[max_pos_row[i]]
    
    new_anndata.obs['cluster_ensemble'] = new_anndata.obs[method_1_name]
    # after_cross = pd.crosstab(new_anndata.obs[method_1_name],new_anndata.obs[method_2_name])
    # print("Before correction:")
    # print(new_anndata_cross)
    # print("After correction:")
    # print(after_cross)
    return(new_anndata)






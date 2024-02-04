# -*- coding: utf-8 -*-
"""
"""
import os
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
import time
from pyod.models.copod import COPOD


########################################################################################
### Algo ensemble by coefficient of variation ##########################################
########################################################################################
def correct_ctab(merge_table, ctab, ctab_column_correct_name):
    max_pos_row = ctab[ctab_column_correct_name].apply(np.argmax, axis=0).values
    max_pos_row_name_index = ctab.index.values[max_pos_row]
    
    tmp_table_mod_row = merge_table.copy()

    for i,j in zip(max_pos_row_name_index, ctab_column_correct_name):
        index = tmp_table_mod_row[ctab.columns.name] == j  # all the sample belongs to some single desc cluster
        tmp_table_mod_row.loc[index, ctab.index.name] = i
    
    tmp_table_mod_row = tmp_table_mod_row.rename(columns={ctab.index.name:'cluster_ensemble'}) 
    return tmp_table_mod_row

def get_correct_name(ctab, cv_threshold):
    cv_column = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=0).values  #apply function to each column.
    cv_row = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=1).values

    ctab_column_correct_name = ctab.columns.values[cv_column >= cv_threshold] # default method cv > 2.0
    ctab_row_correct_name = ctab.index.values[cv_row >= cv_threshold] # default method cv > 2.0
    return(ctab_row_correct_name, ctab_column_correct_name)

def correct_ann_cv(df, method_1_name, method_2_name, cv_threshold):
    merge_df = df[[ method_1_name, method_2_name ]]  
    pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
    ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
    df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)
    df['cluster_ensemble'] = df_new_pred['cluster_ensemble']
    return df

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
            correct_C_ref = correct_ann_cv(temp, 'algo1','algo2',cv_threshold)
            correct_C_asst = correct_ann_cv(temp, 'algo2','algo1',cv_threshold)

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

            if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
                if PRINT_INFO: print('early stop at iter = ',iter)
                break

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









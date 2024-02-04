# -*- coding: utf-8 -*-
import os
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from . import calc_acc
import time
import pdb

### Algo ensemble by coefficient of variation 

def correct_ann_cv_df(anndata_df, method_1_name, method_2_name, cv_threshold):
    merge_df = anndata_df.copy()
    pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
    ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
    df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)

    if 'cluster_ensemble' in anndata_df:
        anndata_df = anndata_df.drop(columns="cluster_ensemble")

    anndata_df.loc[:,['cluster_ensemble']]  = df_new_pred['cluster_ensemble']
    return anndata_df

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


def cell_label_update_CV_df(adata_df_input,algo_name,cv_threshold,ITER_NUM=0,print_info = False):
    adata_df_output = adata_df_input.copy()
    adata_df_calc = adata_df_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    if print_info:
        print('--------------  ITER_NUM',ITER_NUM)
    
    for iter in range(0,ITER_NUM+1):
        if iter==0:
            adata_df_calc.loc[:,['algo1']] = pd.Categorical(adata_df_input[algo_name[0]]) # init C_ref
            df_C_ref = pd.DataFrame(adata_df_input[algo_name[0]])
            adata_df_calc.loc[:,['algo2']] = pd.Categorical(adata_df_input[algo_name[1]]) # init C_asst
        else: # use new result to assit main algo(algo1)
            df_C_ref = pd.DataFrame(adata_df_correct_C_ref['cluster_ensemble'])

        ## Show cross table each iter
        if print_info:
            print(pd.crosstab(adata_df_calc['algo1'],adata_df_calc['algo2']))

        ## input for c_tab
        adata_df_correct_C_ref = correct_ann_cv_df(adata_df_calc, 'algo1','algo2',cv_threshold)
        adata_df_correct_C_asst = correct_ann_cv_df(adata_df_calc, 'algo2','algo1',cv_threshold)

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
            if print_info:
                print('iter = ',iter)
            break

    try:
        print('Total CTEC-DB time = ',time.time()-t_start)
    except:
        pass
    
    return adata_df_output



def get_correct_name(ctab, cv_threshold):
    cv_column = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=0).values  #apply function to each column.
    cv_row = ctab.apply(lambda x: np.std(x)/np.mean(x), axis=1).values

    ctab_column_correct_name = ctab.columns.values[cv_column >= cv_threshold] # default method cv > 2.0
    ctab_row_correct_name = ctab.index.values[cv_row >= cv_threshold] # default method cv > 2.0
    return(ctab_row_correct_name, ctab_column_correct_name)


# def correct_ctab(merge_table, ctab, ctab_column_correct_name):
#     max_pos_row = ctab[ctab_column_correct_name].apply(np.argmax, axis=0).values
#     max_pos_row_name_index = ctab.index.values[max_pos_row]
    
#     tmp_table_mod_row = merge_table.copy()

#     for i,j in zip(max_pos_row_name_index, ctab_column_correct_name):
#         index = tmp_table_mod_row[ctab.columns.name] == j  # all the sample belongs to some single desc cluster
#         tmp_table_mod_row.loc[index, ctab.index.name] = i
    
#     tmp_table_mod_row = tmp_table_mod_row.rename(columns={ctab.index.name:'cluster_ensemble'}) 
#     return tmp_table_mod_row

def correct_ann_cv(anndata, method_1_name, method_2_name, cv_threshold):
    adata_input = anndata.copy()
    merge_df = adata_input.obs[[ method_1_name, method_2_name ]]  
    pre_cross = pd.crosstab(merge_df[ method_1_name], merge_df[ method_2_name])
    ___, ctab_column_correct_name = get_correct_name(pre_cross, cv_threshold = cv_threshold)
    df_new_pred = correct_ctab(merge_df,pre_cross,ctab_column_correct_name)
    adata_input.obs['cluster_ensemble'] = df_new_pred['cluster_ensemble']
    return adata_input

def cell_label_update_CV(adata_input,algo_name,one_gt_name,cv_threshold,ITER_NUM=0):
    adata_output = adata_input.copy()
    adata_calc = adata_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    print('--------------  ITER_NUM',ITER_NUM)
    
    for iter in range(0,ITER_NUM+1):
        if iter==0:
            adata_calc.obs['algo1'] = pd.Categorical(adata_input.obs[algo_name[0]]) # init C_ref
            df_C_ref = pd.DataFrame(adata_input.obs[algo_name[0]])
            adata_calc.obs['algo2'] = pd.Categorical(adata_input.obs[algo_name[1]]) # init C_asst
        else: # use new result to assit main algo(algo1)
            df_C_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])

        ## Show cross table each iter
        print(pd.crosstab(adata_calc.obs['algo1'],adata_calc.obs['algo2']))

        ## input for c_tab
        adata_correct_C_ref = correct_ann_cv(adata_calc, 'algo1','algo2',cv_threshold)
        adata_correct_C_asst = correct_ann_cv(adata_calc, 'algo2','algo1',cv_threshold)

        ## show ARI, NMI if possible
        adata_output.obs['cluster_ensemble'] = adata_correct_C_ref.obs['cluster_ensemble']
        if one_gt_name!=None:
            df = adata_output.obs[[ 'cluster_ensemble',one_gt_name]]        
            # print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble'))

        # update result for next iteration
        adata_calc.obs['algo1'] = adata_correct_C_ref.obs['cluster_ensemble'] # update C'ref
        adata_calc.obs['algo2'] = adata_correct_C_asst.obs['cluster_ensemble'] # update C'asst

        df_C_prime_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])
        aa = np.array(df_C_prime_ref.iloc[:,0].apply(str))
        bb = np.array(df_C_ref.iloc[:,0].apply(str))
        case_same = np.where(aa==bb)[0].shape[0]
        case_total = df_C_prime_ref.shape[0]
        
        if one_gt_name!=None:
            print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble',decimals=3),'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
        else:
            print('iter = ',iter,'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))

        ratio_before_after.append(1.0*case_same/case_total)
        if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
            print('iter = ',iter)
            break

    try:
        print('Total CTEC-DB time = ',time.time()-t_start)
    except:
        pass

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

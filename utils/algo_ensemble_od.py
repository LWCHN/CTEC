# -*- coding: utf-8 -*-
import os
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from . import calc_acc
from pyod.models.copod import COPOD
import time
import pdb

### Algo ensemble by outlier detection

def cell_label_update_OD_df(adata_df_input,adata_PCA,algo_name,ITER_NUM=0,print_info = False):
    adata_df_output = adata_df_input.copy()
    adata_df_calc = adata_df_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    if print_info:
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
        if print_info:
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
        if print_info:
            print('iter = ',iter,'; ratio_before_after = ',np.round(1.0*case_same/case_total, decimals=5))
        
        ratio_before_after.append(1.0*case_same/case_total)  
        if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
            if print_info:
                print('iter = ',iter)
            break

    try:
        print('Total CTEC-OB time = ',time.time()-t_start)
    except:
        pass
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





"""
def cell_label_update_OD(adata_input,algo_name,one_gt_name,ITER_NUM=0):
    adata_output = adata_input.copy()
    adata_calc = adata_input.copy()
    t_start=time.time()
    ratio_before_after = list()
    ITER_NUM=ITER_NUM
    print('--------------  ITER_NUM total= ',ITER_NUM)
    
    for iter in range(0,ITER_NUM+1):
        if iter==0:
            adata_calc.obs['algo1'] = adata_input.obs[algo_name[0]] # init C_ref
            df_C_ref = pd.DataFrame(adata_input.obs[algo_name[0]])
            adata_calc.obs['algo2'] = adata_input.obs[algo_name[1]] # init C_asst
            del adata_input
        else: # use new result to assit main algo(algo1)
            df_C_ref = pd.DataFrame(adata_correct_C_ref.obs['cluster_ensemble'])
        
        ## Show cross table each iter
        print(pd.crosstab(adata_calc.obs['algo1'],adata_calc.obs['algo2']))

        ## input for c_tab
        adata_correct_C_ref = correct_ann_by_pca(adata_calc, 'algo1','algo2')
        adata_correct_C_asst = correct_ann_by_pca(adata_calc, 'algo2','algo1')

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
        ratio_before_after.append(1.0*case_same/case_total)
        
        if one_gt_name!=None:
            print('iter = ',iter, calc_acc.calc_all_acc_simple(df,gt_name=one_gt_name,pred_name='cluster_ensemble',decimals=3), '; ratio_before_after = ', np.round(ratio_before_after[-1], decimals=5))
        else:
            print('iter = ',iter, '; ratio_before_after = ', np.round(ratio_before_after[-1], decimals=5))
            
        if len(ratio_before_after) >=2 and (ratio_before_after[-1] == ratio_before_after[-2]): 
            print('early stop at iter = ',iter)
            break

    try:
        print('Total CTEC-OB time = ',time.time()-t_start)
    except:
        pass
    return adata_output



def correct_ann_by_pca(anndata, method_1_name, method_2_name):
    new_anndata = anndata.copy()
    del anndata
    new_anndata.obs[method_1_name] = new_anndata.obs[method_1_name].astype(str).astype("category")
    new_anndata.obs[method_2_name] = new_anndata.obs[method_2_name].astype(str).astype("category")

    new_anndata_cross = pd.crosstab(new_anndata.obs[method_1_name],new_anndata.obs[method_2_name])

    max_pos_row = new_anndata_cross.apply(np.argmax, axis=0).values
    assert len(max_pos_row) == new_anndata_cross.shape[1], "please check you max position"

    for i in range(len(max_pos_row)):
        train_index = np.logical_and(new_anndata.obs[method_2_name] == new_anndata_cross.columns[i],
                                     new_anndata.obs[method_1_name] == new_anndata_cross.index.values[max_pos_row[i]])
        
        print('train_index = ',train_index)
        aa=new_anndata.obsm["X_pca"]
        print('aa[train_index] = ',aa[train_index])

        
        tmp_train = new_anndata[train_index].obsm["X_pca"] # ！！！mod
        print('tmp_train = ',tmp_train)

        if (aa[train_index]==tmp_train).all():
            print('aa==bb all')
        #tmp_train = new_anndata[train_index].X

        test_index = np.logical_and(new_anndata.obs[method_2_name] == new_anndata_cross.columns[i],
                                    new_anndata.obs[method_1_name] != new_anndata_cross.index.values[max_pos_row[i]])
        tmp_test = new_anndata[test_index].obsm["X_pca"] # ！！！mod
        #tmp_test = new_anndata[test_index].X
        if (aa[test_index]==tmp_test).all():
            print("(aa[test_index]==tmp_test).all()")
        pdb.set_trace()
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
    return(new_anndata)
"""








# -*- coding: utf-8 -*-
## import system file
from operator import is_not
import os
import scanpy as sc
import numpy as np
import time

from utils.algo_ctec_multiple import CTEC_OB_multiple, CTEC_DB_multiple
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from utils import calc_acc
import pandas as pd
import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

from utils.algo_ctec import post_process_cluster_name_with_order


def evaluate_df(df_input,pred_name,GTname):
    df = df_input[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_acc.calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name

def evaluate(adata,pred_name,GTname):
    df = adata.obs[[pred_name,GTname]]
    acc_results_value, acc_results_name= calc_acc.calc_all_acc_simple(df,GTname,pred_name,decimals=4)
    return acc_results_value, acc_results_name


if __name__ == '__main__':
    WORK_PATH = '/ctec_work'
    DATA_PATH = os.path.join(WORK_PATH,'base_method_result/multi_method')
    SAVE_PATH = WORK_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='demo', required=False, help='experiment id by int')
    args = parser.parse_args()    
    EXP = args.exp_id
    exist_ground_truth_cell_type=1

    save_path = os.path.join(SAVE_PATH,'Result_Ensemble_Multi_Method_Exp_'+str(EXP))
    try:
        os.mkdir(save_path)
    except:
        pass



    data_list_fullname = ['macaque_bc', 'human_pbmc_GSE96583', 'mouse_cortex_SCP425', 'human_pancreas',  'paul15']
    algo_name_raw = ['SC3','CIDR', 'Seurat', 'Kmeans', 'SIMLR']


    for i in range(0,len(data_list_fullname)):
        data_name = data_list_fullname[i]
        algo_name = algo_name_raw[:] #deep copy list
        print('\n\n','='*100)
        print('data_name=',data_name,', EXP=',EXP)
        

        #[1.1 read data]
        adata = sc.read(os.path.join(DATA_PATH,data_list_fullname[i]+'_result.h5ad'))


        GTname = 'celltype'
        ADATA_RAW = adata.copy()

        algo_name = algo_name_raw[:] #deep copy list
        print('algo list ** BEFORE** filtering:\n', algo_name)
        algo_score_list = []
        for algo in algo_name:
            ch_score = calinski_harabasz_score(adata.obsm['X_pca'], adata.obs[algo])
            algo_score_list.append((algo, ch_score))
        algo_score_list.sort(key=lambda x: x[1], reverse=False)
        algo_name.remove(algo_score_list[0][0]) # remove the worse algo based on CH score
        print('algo list ** AFTER ** filtering:\n', algo_name)

        num_algo = len(algo_name_raw)
        print('running CTEC_DB')
        CV_THRESHOLD = 0.5
        ITER_NUMBER = 10
        t0=time.time()
        adata_ctec_db = CTEC_DB_multiple(ADATA_RAW,algo_name,CV_THRESHOLD, ITER_NUMBER,PRINT_INFO=True)
        time_ctec_db=time.time()-t0
        adata_ctec_db = post_process_cluster_name_with_order(adata_ctec_db, 'cluster_ensemble')

        adata_ctec_db.obs['cluster_ensemble'] = pd.Categorical( adata_ctec_db.obs['cluster_ensemble'].astype(str) )
        ADATA_RAW.obs['ctec_db_labels'] = adata_ctec_db.obs['cluster_ensemble']
        ADATA_RAW.obs.to_csv(os.path.join(save_path, '{}_processed_CTECDB_{}algo.csv'.format(data_name, num_algo)))

        if exist_ground_truth_cell_type: 
            acc_ctec_db, acc_results_name = evaluate(ADATA_RAW,'ctec_db_labels',GTname)
            print('\n\n____________acc: ctec_db_labels')
            print(acc_results_name,acc_ctec_db)
            result_acc_path = os.path.join(save_path,'result_acc=={}==_CTECDB_{}algo.csv'.format(data_name, num_algo))
            with open(result_acc_path  ,'a+') as f:
                f.write('File,'+data_name+'\n')
                f.write('ITER_NUMBER,'+str(ITER_NUMBER)+'\n')
                f.write('Total_time,'+str(time_ctec_db)+'\n')
                f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
                f.write('Acc_ctec_db,'+calc_acc.list2str(acc_ctec_db)+','+'NA ,'+str(time_ctec_db)+'\n')
                f.write('\n')
        else:
            pass



        print('\n running CTEC_OB')
        ITER_NUMBER = 10
        t0=time.time()
        adata_ctec_ob = CTEC_OB_multiple(ADATA_RAW,algo_name,ITER_NUMBER,PRINT_INFO=True)
        time_ctec_ob=time.time()-t0
        adata_ctec_ob = post_process_cluster_name_with_order(adata_ctec_ob, 'cluster_ensemble')
        
        adata_ctec_ob.obs['cluster_ensemble'] = pd.Categorical( adata_ctec_ob.obs['cluster_ensemble'].astype(str) )
        ADATA_RAW.obs['ctec_ob_labels'] = adata_ctec_ob.obs['cluster_ensemble']
        ADATA_RAW.obs.to_csv((os.path.join(save_path, '{}_processed_CTECOB_{}algo.csv'.format(data_name, num_algo))))
    
        if exist_ground_truth_cell_type:
            acc_ctec_ob, acc_results_name = evaluate(ADATA_RAW,'ctec_ob_labels',GTname)
            print('\n\n____________acc: ctec_ob_labels')
            print(acc_results_name,acc_ctec_ob)
            result_acc_path = os.path.join(save_path,'result_acc=={}==_CTECOB_{}algo.csv'.format(data_name, num_algo))
            with open(result_acc_path  ,'a+') as f:
                f.write('File,'+data_name+'\n')
                f.write('ITER_NUMBER,'+str(ITER_NUMBER)+'\n')
                f.write('Total_time,'+str(time_ctec_ob)+'\n')
                f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'resolution'+','+'time_consuming'+'\n')
                f.write('Acc_ctec_ob,'+calc_acc.list2str(acc_ctec_ob)+','+'NA ,'+str(time_ctec_ob)+'\n')
                f.write('\n')
        else:
            pass
        

        
        # -------------------------------------------------------------------------------
        ## acc summary
        adata_raw_obs_df = pd.DataFrame(ADATA_RAW.obs)
        ground_truth_df = adata_raw_obs_df[GTname].astype(str) 
        algo_ctec_ob_df = adata_raw_obs_df['ctec_ob_labels'].astype(str) 
        algo_ctec_db_df = adata_raw_obs_df['ctec_db_labels'].astype(str) 

        algo1_df = adata_raw_obs_df[algo_name_raw[0]].astype(str)
        algo2_df = adata_raw_obs_df[algo_name_raw[1]].astype(str)
        algo3_df = adata_raw_obs_df[algo_name_raw[2]].astype(str)
        algo4_df = adata_raw_obs_df[algo_name_raw[3]].astype(str)
        algo5_df = adata_raw_obs_df[algo_name_raw[4]].astype(str)
        SAFE_df = adata_raw_obs_df['SAFE'].astype(str)
        SAME_df = adata_raw_obs_df['SAME'].astype(str)


        print('\n---- Summary ----')
        DF_FOR_ACC1 = pd.concat([algo1_df, ground_truth_df], axis=1, join="inner")
        acc_algo1, acc_results_name = evaluate_df(DF_FOR_ACC1,algo_name_raw[0],GTname)
        print(acc_results_name,acc_algo1,algo_name_raw[0])

        DF_FOR_ACC2 = pd.concat([algo2_df, ground_truth_df], axis=1, join="inner")
        acc_algo2, acc_results_name = evaluate_df(DF_FOR_ACC2,algo_name_raw[1],GTname)
        print(acc_results_name,acc_algo2,algo_name_raw[1])

        DF_FOR_ACC3 = pd.concat([algo3_df, ground_truth_df], axis=1, join="inner")
        acc_algo3, acc_results_name = evaluate_df(DF_FOR_ACC3,algo_name_raw[2],GTname)
        print(acc_results_name,acc_algo3,algo_name_raw[2])

        DF_FOR_ACC4 = pd.concat([algo4_df, ground_truth_df], axis=1, join="inner")
        acc_algo4, acc_results_name = evaluate_df(DF_FOR_ACC4,algo_name_raw[3],GTname)
        print(acc_results_name,acc_algo4,algo_name_raw[3])

        DF_FOR_ACC5 = pd.concat([algo5_df, ground_truth_df], axis=1, join="inner")
        acc_algo5, acc_results_name = evaluate_df(DF_FOR_ACC5,algo_name_raw[4],GTname)
        print(acc_results_name,acc_algo5,algo_name_raw[4])

        DF_FOR_SAFE = pd.concat([SAFE_df, ground_truth_df], axis=1, join="inner")
        acc_SAFE, acc_results_name = evaluate_df(DF_FOR_SAFE,'SAFE',GTname)
        print(acc_results_name,acc_SAFE,'SAFE')

        DF_FOR_SAME = pd.concat([SAME_df, ground_truth_df], axis=1, join="inner")
        acc_SAME, acc_results_name = evaluate_df(DF_FOR_SAME,'SAME',GTname)
        print(acc_results_name,acc_SAME,'SAME')

        DF_FOR_ACC_ctec_ob = pd.concat([algo_ctec_ob_df, ground_truth_df], axis=1, join="inner")
        acc_ctec_ob, acc_results_name = evaluate_df(DF_FOR_ACC_ctec_ob,'ctec_ob_labels',GTname)
        print(acc_results_name,acc_ctec_ob,'CTEC_OB')
        
        DF_FOR_ACC_ctec_db = pd.concat([algo_ctec_db_df, ground_truth_df], axis=1, join="inner")
        acc_ctec_db, acc_results_name = evaluate_df(DF_FOR_ACC_ctec_db,'ctec_db_labels',GTname)
        print(acc_results_name,acc_ctec_db,'CTEC_DB')

        result_acc_path = os.path.join(save_path,'result_acc=='+data_name+'==_CTEC_multiple.csv')
        with open(result_acc_path  ,'a+') as f:
            f.write('File,'+data_name+'\n')
            f.write('Acc_Name,'+calc_acc.list2str(acc_results_name)+','+'time_consuming'+'\n')
            f.write('Acc_'+algo_name_raw[0]+','+calc_acc.list2str(acc_algo1)+','+str('')+'\n')
            f.write('Acc_'+algo_name_raw[1]+','+calc_acc.list2str(acc_algo2)+','+str('')+'\n')
            f.write('Acc_'+algo_name_raw[2]+','+calc_acc.list2str(acc_algo3)+','+str('')+'\n')
            f.write('Acc_'+algo_name_raw[3]+','+calc_acc.list2str(acc_algo4)+','+str('')+'\n')
            f.write('Acc_'+algo_name_raw[4]+','+calc_acc.list2str(acc_algo5)+','+str('')+'\n')
            f.write('Acc_'+'SAFE'+','+calc_acc.list2str(acc_SAFE)+','+str('')+'\n')
            f.write('Acc_'+'SAME'+','+calc_acc.list2str(acc_SAME)+','+str('')+'\n')
            f.write('Acc_CTEC_OB,'+calc_acc.list2str(acc_ctec_ob)+','+str(time_ctec_ob)+'\n')
            f.write('Acc_CTEC_DB,'+calc_acc.list2str(acc_ctec_db)+','+str(time_ctec_db)+'\n')
            f.write('\n')



        del ADATA_RAW
        del adata
        gc.collect()
        print('data_name_NUM=',i,', EXP=',EXP)
        print('='*100)
    
    

def draw_umap(i):
    num_algo = len(algo_name_raw)
    data_name = data_list_fullname[i]
    print('\n\n','='*100)
    print('data_name=',data_name,', EXP=',EXP)
    

    #[1.1 read data]
    adata = sc.read(os.path.join(DATA_PATH,data_list_fullname[i]+'_res.h5ad'))       
    GTname = 'celltype'



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

    # data_name, data_ext,GTname,class_nums,BATCH_KEY,N_HIGH_VAR,res_leiden,res_louvain = loaddata_default(DATASET_NUM)
    print('\n\n','='*100)
    print('Step 7: calc UMAP')
    print( 'EXP=',EXP)
    print('data_name = ',data_name)
    print('='*100)
    t_start_dataset = time.time()


    ## load dataframe CTEC-DB
    PATH_ctec_db_labels_df  = os.path.join(save_path, '{}_processed_CTECDB_{}algo.csv'.format(data_name, num_algo))
    ctec_db_labels_df = pd.read_csv(PATH_ctec_db_labels_df,index_col=0)

    ## load dataframe CTEC-OB
    PATH_ctec_ob_labels_df = os.path.join(save_path, '{}_processed_CTECOB_{}algo.csv'.format(data_name, num_algo))
    ctec_ob_labels_df = pd.read_csv(PATH_ctec_ob_labels_df,index_col=0) 

    plot_obsname = [GTname,algo_name_raw[0],algo_name_raw[1],algo_name_raw[2],algo_name_raw[3],algo_name_raw[4],'SAFE','SAME','ctec_db_labels','ctec_ob_labels']
    plot_showname = ['Reference standard annotation','SC3','CIDR', 'Seurat', 'Kmeans', 'SIMLR','SAFE','SAME','CTEC-DB','CTEC-OB']
    save_name = data_name
    SAVE_PDF = 1

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

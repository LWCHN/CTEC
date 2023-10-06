# -*- coding: utf-8 -*-
"""
"""
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
from utils.calc_acc import evaluate_df
import pandas as pd
import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

from utils.algo_ctec import post_process_cluster_name_with_order



WORK_PATH = './'
DATA_PATH = os.path.join(WORK_PATH,'base_method_result')
SAVE_PATH = WORK_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=int, default=2, required=False, help='experiment id by int')
args = parser.parse_args()    
EXP = args.exp_id
exist_ground_truth_cell_type=1

save_path = os.path.join(SAVE_PATH,'Result_Ensemble_Multi_Method_'+str(EXP))
try:
    os.mkdir(save_path)
except:
    pass


data_list_fullname = ['macaque_bc', 'human_pbmc_GSE96583', 'mouse_cortex_SCP425', 'human_pancreas',  'paul15']  
algo_name_raw = ['SC3','CIDR', 'Seurat', 'tsne', 'SIMLR']


for i in range(0, len(data_list_fullname)):
    num_algo = len(algo_name_raw)
    data_name = data_list_fullname[i]
    print('\n\n','='*100)
    print('data_name=',data_name,', EXP=',EXP)
    

    #[read data]
    adata = sc.read(os.path.join(DATA_PATH,data_list_fullname[i]+'_res.h5ad'))
    sc.tl.pca(adata, svd_solver='arpack',random_state=1) # use default paras
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_pca", random_state=1) # use default paras

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
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_OnData.pdf'),bbox_inches='tight',dpi=FIG_DPI)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_OnData.png'),bbox_inches='tight',dpi=FIG_DPI)
    with plt.rc_context():  # Use this to set figure params like size and dpi        
        sc.pl.umap(adata,
                color=group_name,
                palette=colordict_gt,
                legend_loc = 'right margin',
                legend_fontsize = 'xx-small',
                legend_fontoutline=2)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_Right.pdf'),bbox_inches='tight',dpi=FIG_DPI)
        plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_Right.png'),bbox_inches='tight',dpi=FIG_DPI)


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
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_Right.pdf'),bbox_inches='tight',dpi=FIG_DPI)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_Right.png'),bbox_inches='tight',dpi=FIG_DPI)
        with plt.rc_context():  # Use this to set figure params like size and dpi        
            sc.pl.umap(adata,
                    color=group_name,
                    palette=colordict_gp,
                    legend_loc = 'on data',
                    legend_fontsize = 'xx-small',
                    legend_fontoutline=2)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_OnData.pdf'),bbox_inches='tight',dpi=FIG_DPI)
            plt.savefig(os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_'+str(num_algo)+'Algo_OnData.png'),bbox_inches='tight',dpi=FIG_DPI)
        print('umap image Saved:'+ os.path.join(save_path_umap, data_name+'_'+group_name+'_UMAP_OnData......'))

    print('total calc time = ',time.time()-t0)

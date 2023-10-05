# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

def match_cluster_name(ctab):
    dict_gp_to_gt = dict()
    dict_gt_to_gp = dict()
    temp_list_match_gt = list()
    temp_list_match_gp = list()
    for i in range(0,ctab.shape[0]): 
        ## based on gt(row) find pred (col)
        one_gt_name = ctab.index[i]
        max_in_row =  ctab.iloc[i,:].values.max()
        col_index_max = np.where(ctab.iloc[i,:].values==max_in_row)[0][0]
        one_gp_name = ctab.columns[col_index_max]

        ## based on best col(pred), double check gt(row)
        one_best_col_max = ctab.iloc[:,col_index_max].values.max()
        
        # print('one_gt_name, max_in_row = ',one_gt_name,max_in_row)

        if max_in_row==one_best_col_max:
            dict_gt_to_gp = update_corr_dict(dict_gt_to_gp,one_gt_name, max_in_row, one_gp_name)
            temp_list_match_gt.append(one_gt_name)
            
            # print('dict_gp_to_gt = ',dict_gp_to_gt)
            # print('temp_list_match_gt = ',temp_list_match_gt)
            # print("dict_gt_to_gp = ",dict_gt_to_gp)
        

        else:

            row_index_best_gt = np.where(ctab.iloc[:,col_index_max].values==one_best_col_max)[0][0]
            new_best_gt_name = ctab.index[row_index_best_gt]
            temp_list_match_gt.append(new_best_gt_name)


            if one_best_col_max < ctab.iloc[row_index_best_gt,:].max():

                better_best_gt_name = one_gt_name
                dict_gt_to_gp = update_corr_dict(dict_gt_to_gp,better_best_gt_name, max_in_row, one_gp_name)
            else:
                dict_gt_to_gp = update_corr_dict(dict_gt_to_gp,new_best_gt_name, one_best_col_max, one_gp_name)

            # print('row_index_best_gt,new_best_gt_name = ',row_index_best_gt,new_best_gt_name)            
            # print('temp_list_match_gt = ',temp_list_match_gt)
            # print("dict_gt_to_gp = ",dict_gt_to_gp)

    dict_gp_to_gt = convert_dict_gt_gp(dict_gt_to_gp)
    # print("dict_gp_to_gt = ",dict_gp_to_gt)
    return dict_gp_to_gt

def convert_dict_gt_gp(dict_gt_to_gp):
    list1_gt_name=list()
    list2_gp_name=list()
    list3_gp_quantity=list()
    for key, value in dict_gt_to_gp.items():
        list1_gt_name.append(key)
        list2_gp_name.append(dict_gt_to_gp[key][0])
        list3_gp_quantity.append(dict_gt_to_gp[key][1])
    
    # print(list1_gt_name)
    # print(list2_gp_name)
    # print(list3_gp_quantity)
    dict_gp_to_gt=dict()
    
    # get max quantity
    max_quantity=dict()
    for i in range(0,len(list2_gp_name)):
        new_key = list2_gp_name[i]
        quantity = list3_gp_quantity[i]
        if not new_key in max_quantity.keys():
            max_quantity[new_key] =quantity
        else:
            if quantity > max_quantity[new_key]:
                max_quantity[new_key] = quantity
    # print('max_quantity = ',max_quantity)       

    for i in range(0,len(list2_gp_name)):
        new_key = list2_gp_name[i]
        new_value = list1_gt_name[i]
        quantity = list3_gp_quantity[i]

        if not new_key in dict_gp_to_gt.keys():
            dict_gp_to_gt[new_key] = new_value

        elif quantity == max_quantity[new_key]:
            dict_gp_to_gt[new_key] = new_value
    return dict_gp_to_gt

def  update_corr_dict(mydict,new_key_gt, new_value, cluster_name):

    if not new_key_gt in mydict.keys():
        mydict[new_key_gt] = [str(cluster_name),  new_value]

    else:
        if new_value > mydict[new_key_gt][1]:
            mydict[new_key_gt] = [str(cluster_name),  new_value]
    return mydict




def give_color_to_group(name_list_gp,dict_gp_to_gt,full_color_list,colordict_gt):
    # give color to group name
    # firstly make dict color list for corresponding cluster
    # print('firstly make dict color list for corresponding cluster')
    output_colordict_gp = dict()
    for ele in name_list_gp:
        ele = str(ele)
        if ele in dict_gp_to_gt.keys():
            output_colordict_gp[ele] = str(colordict_gt[dict_gp_to_gt[ele]])
            # print("output_colordict_gp["+str(ele)+"] = ", output_colordict_gp[ele] )
            full_color_list.remove(output_colordict_gp[ele])
    # secondly, make dict color for other cluster in group 
    # print('secondly, make dict color for other cluster in group ')
    for ele in name_list_gp:
        ele = str(ele)
        if not ele in output_colordict_gp.keys():
            one_color = full_color_list[0]
            output_colordict_gp[ele] = one_color
            full_color_list.remove(one_color)
            # print("output_colordict_gp["+str(ele)+"] = ", output_colordict_gp[ele] )
    # print(output_colordict_gp)
    return output_colordict_gp

def get_color_dict(adata_raw,name_gt,name_gp,colordict_gt,FULLCOLOR_LIST):
    full_color_list = FULLCOLOR_LIST.copy()
    df12 = adata_raw.obs[[name_gt, name_gp ]]  
    ctab = pd.crosstab(df12[name_gt], df12[name_gp])
    # print('ctab = \n',ctab)
    name_list_gp = list(set(adata_raw.obs[name_gp]))
    name_list_gt = list(set(adata_raw.obs[name_gt]))
    
    dict_gp_to_gt= match_cluster_name(ctab)
    # print('dict_gp_to_gt = ',dict_gp_to_gt)
    # print('name_list_gt = ',name_list_gt)

    return give_color_to_group(name_list_gp,dict_gp_to_gt,full_color_list,colordict_gt)



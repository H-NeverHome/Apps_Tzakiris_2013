# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:58:54 2021

@author: de_hauk
"""
### Process raw data under path, nothin prestored

import pandas as pd
import glob
import numpy as np
from autoimpute.imputations import MultipleImputer
data_path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
ground_truth_file = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
all_files = glob.glob(data_path + "/*.csv")

SAMPLE_fullinfo = pd.read_csv(ground_truth_file).drop(columns = ['Unnamed: 0']).copy()  

unq_id = []
DATA_raw_DF = pd.DataFrame()

final_dat_A = {}
final_dat_B = {}
for data in all_files:
    data_ID = pd.DataFrame()
    unique_ID = data[-12] + '_' + data[-5] 
    unq_id.append(unique_ID)
    curr_raw_data = pd.read_csv(data,header=None, sep='\t').drop(axis='index', labels = [0,1])[2]
    perspective = []
    answer_correct = [] #1 yes // 0 no
    answer_raw = [] # YES -> 1 // NO -> 0 // Do you remember face?
    for data_point in curr_raw_data:
        if len(data_point) < 5:
            perspective.append(data_point)
        if ('1' in data_point[0])and (len(data_point)>5):
            answer_correct.append(1)
        elif ('0' in data_point[0])and (len(data_point)>5):
            answer_correct.append(0)
        if '[YES]' in data_point:
            answer_raw.append(1)
        elif '[NO]' in data_point:
            answer_raw.append(0)
        elif 'missed' in data_point:
            answer_raw.append(np.nan)
    view_dep_stim_ID = {}       
    for VD_id,VD_num_ID in zip(np.unique(perspective),range(len(np.unique(perspective)))):
        view_dep_stim_ID[VD_id] = VD_num_ID
    view_dep_L = []
    for i in perspective:
        view_dep_L.append(view_dep_stim_ID[i])
    
    check_L = len(np.unique(perspective)) == len(np.unique(view_dep_L))
        
    data_ID['perspective'] = view_dep_L
    data_ID['perf'] = answer_correct
    data_ID['answer'] = answer_raw
    data_ID['stim_IDs_VI'] = SAMPLE_fullinfo['stim_IDs']
    data_ID['new_IDs'] = SAMPLE_fullinfo['new_IDs']
    data_ID['stim_IDs_VD'] = view_dep_L
    
    data_id_fin = data_ID.copy().loc[data_ID['answer'].isna()==False].reset_index(drop=True)
    
    n_prev_VI = []
    n_prev_VD = []
    unq_stim_ID_VI = {}
    unq_stim_ID_VD = {}
    for stim_VI,stim_VD in zip(data_id_fin['stim_IDs_VI'], data_id_fin['stim_IDs_VD']):

        key_VI, key_VD = str(stim_VI),str(stim_VD)
        
        ### count VI_Stims
        curr_keys_VI = [i for i in unq_stim_ID_VI.keys()]
        
        if key_VI not in curr_keys_VI:
            unq_stim_ID_VI[key_VI] = 1
        elif key_VI in curr_keys_VI:
            unq_stim_ID_VI[key_VI] = unq_stim_ID_VI[key_VI]+1
        
        n_prev_VI.append(unq_stim_ID_VI[key_VI])
        
        ## count VD_Stims
        curr_keys_VD = [i for i in unq_stim_ID_VD.keys()]

        if key_VD not in curr_keys_VD:
            unq_stim_ID_VD[key_VD] = 1
        elif key_VD in curr_keys_VD:
            unq_stim_ID_VD[key_VD] = unq_stim_ID_VD[key_VD]+1
        n_prev_VD.append(unq_stim_ID_VD[key_VD])  
        
    data_id_fin['n_prev_VI'] = n_prev_VI
    data_id_fin['n_prev_VD'] = n_prev_VD
    
    if 'A' in unique_ID:
        final_dat_A[unique_ID] = data_id_fin
    elif 'B' in unique_ID:
        final_dat_B[unique_ID] = data_id_fin
    final_dat = {'A':final_dat_A,
                 'B':final_dat_B}
    
    
    
    
    
    
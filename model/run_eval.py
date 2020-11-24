# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:48:28 2020

@author: de_hauk
"""

from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT
from class_import import fit_data_noCV_irr_len_data
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 


########### Get Data

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
#data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

data_1_sample_raw = data_old_sample(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\A_T_Implementation\data_lab_jansen')

##### Reformat data

##### Reformat data
ref_dat_raw = data_2_sample['raw_data']
unq_ids = data_2_sample['unique_ID']


data_dict_total = {}
data_dict_t1 = {}
data_dict_t2 = {}
for ids in unq_ids:
    curr_df_raw = ref_dat_raw[[i for i in ref_dat_raw if ids in i]].copy()
    curr_df_raw['stim_IDs'] = list(data_2_sample['stat_ground-truth']['stim_IDs'])
    curr_df_raw['new_IDs'] = list(data_2_sample['stat_ground-truth']['new_IDs'])
    curr_df_raw['n_prev_pres'] = list(data_2_sample['stat_ground-truth']['number_of_prev_presentations_raw '])
    
    curr_df = curr_df_raw.loc[curr_df_raw[ids+'_answer'].isna() == False].copy()
    curr_df_1 = curr_df.reset_index(drop=True, inplace=False)
    data_dict_total[ids] = curr_df_1
    if 'A' in ids:
        data_dict_t1[ids] = curr_df_1
    elif 'B' in ids:
        data_dict_t2[ids] = curr_df_1


########### Get Behavioral Performance
task_performance = get_behavioral_performance(data_2_sample)


########### Comp. Modeling

#get results from Apps&Tzakiris 2013

at_model = model_selection_AT()

# For intepretation see https://www.nicebread.de/a-short-taxonomy-of-bayes-factors/

# # fit data sample N=10
# fit_data_sample_1 = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)

# ##### fit data sample N=3, T1 & T2 SEPERATE // NO LOOCV // No imput

fit_data_sample_T1 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, False)
fit_data_sample_T2 = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, False)

# ##### fit data sample N=3, T1 & T2 COMBINED // NO LOOCV // No imput








# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:39:13 2020

@author: de_hauk
"""


from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT,fit_data_noCV_irr_len_data
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 


########### Get Data

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
##### data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

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
    
    curr_df = curr_df_raw.loc[curr_df_raw[ids+'_answer'].isna() == False]
    data_dict_total[ids] = curr_df
    if 'A' in ids:
        data_dict_t1[ids] = curr_df
    elif 'B' in ids:
        data_dict_t2[ids] = curr_df

##### fit data sample N=3, T1 & T2
fit_data_sample_2 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, False)


#fit_data_sample_2 = fit_data_noCV(data_2_sample['imputed_data'], 0.01, False)

########### Get Behavioral Performance
task_performance = get_behavioral_performance(data_2_sample)
# raw_dat = task_performance['used_data']['raw_data']
# perc_corr_1 = raw_dat['2_A_perf'].sum()
# perc_corr_2 = raw_dat['3_A_perf'].sum()



##### T1
task_t1 = task_performance['behavioral_results'].T[[i for i in task_performance['behavioral_results'].T if 'A' in i]]

##### T2
task_t2 = task_performance['behavioral_results'].T[[i for i in task_performance['behavioral_results'].T if 'B' in i]]

########### Comp. Modeling

##### T0 get results from Apps&Tzakiris 2013

at_model = model_selection_AT()

##### T1


##### T2


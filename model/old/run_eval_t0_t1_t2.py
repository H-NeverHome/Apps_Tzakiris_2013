# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:39:13 2020

@author: de_hauk
"""


from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV,fit_data_CV
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
    
    curr_df = curr_df_raw.loc[curr_df_raw[ids+'_answer'].isna() == False].copy()
    curr_df_1 = curr_df.reset_index(drop=True, inplace=False)
    data_dict_total[ids] = curr_df_1
    if 'A' in ids:
        data_dict_t1[ids] = curr_df_1
    elif 'B' in ids:
        data_dict_t2[ids] = curr_df_1

# ##### fit data sample N=3, T1 & T2 // NO LOOCV

fit_data_sample_T1 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, False)
fit_data_sample_T2 = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, False)
fit_data_sample_t1_t2_comb = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, False
# aaaa = fit_data_CV(data_dict_t1, 0.01, False)






# ##### fit data sample N=3, T1 & T2 // WITH LOOCV
# import winsound
# duration = 1000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)
# ########## LLOCV


# #func calls & rand starts
# from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_INDIPENDENTxCONTEXT_CV
# from tqdm import tqdm
# from functools import partial
# from scipy import optimize

# # for ID in collected ids
# #do

# vpn = '1_A'
# curr_data_vpn = data_dict_t1[vpn].copy()

# cv_score = []

# finito = False
   
# for indx in tqdm(range(curr_data_vpn.shape[0])):
#     #print(indx)
#     # holdout data
#     holdout_data = curr_data_vpn.copy().loc[indx]
    
    
    
#     # opt data for model
#     test_data = curr_data_vpn.copy().drop(axis=0,index = indx)
    
    
#     stim_IDs = test_data['stim_IDs'] #stimulus IDs of winning model 
#     new_ID = test_data['new_IDs'] #trials where new ID is introduced 
#     numb_prev_presentations = test_data['n_prev_pres'] #number_of_prev_presentations
#     stim_IDs_perspective = test_data[vpn+'_perspective'] #view dependent
#     VPN_output = test_data[vpn+'_answer'].copy() #VPN answers
#     verbose = False
    
#     parameter_est = {}
    
#     ##### Model Optim
    
#     i=vpn
#     ###### 'VIEW_INDIPENDENTxCONTEXT'
#     data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
#     bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
#     part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
#     res1 = optimize.fmin_l_bfgs_b(part_func_M1,
#                                   approx_grad = True,
#                                   bounds = bounds_M1, 
#                                   x0 = [0.5 for i in range(len(bounds_M1))],
#                                   epsilon=0.01)
#     parameter_est['VIEW_INDIPENDENTxCONTEXT'] = res1[0]
#     data_M1[-1] = True 
#     data_M1_debug = data_M1.copy()
#     params_m_1 = res1[0]
#     m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
#     action = holdout_data[vpn+'_answer']
#     init_V = 0
#     init_C = 0
#     if indx == 0:
#         init_V += m_1[1]['init_val']['init_v']
#         init_C += m_1[1]['init_val']['init_c']
#     else:
#         data_cv_score = m_1[1]['data_store_1'].loc[indx-1]
#         init_V += data_cv_score['history_V']
#         init_C += data_cv_score['history_C']
    
#     #(params,old_vfam,old_cfam,action)
#     cv_trial = VIEW_INDIPENDENTxCONTEXT_CV(params_m_1,init_V,init_C,action)
#     cv_score.append((action,cv_trial))

# fin = np.sum(cv_score)


'''


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
'''

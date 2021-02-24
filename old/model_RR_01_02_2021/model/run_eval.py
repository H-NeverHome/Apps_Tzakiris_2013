# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:48:28 2020

@author: de_hauk
"""
import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT
from class_import import fit_data_noCV_irr_len_data,data_fit_t1_t2_comb
#from class_import import fit_data_NUTS
from class_import import bayes_RFX_cond,orig_procedure
from class_import import reformat_data_within_T,fit_data_CV,bic_modelselect
from class_import import task_rel,corr_lr_func,fit_data_CV_mult
import matlab.engine
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 
from joblib import Parallel, delayed


########### Get Data

folder_path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\processed'
#data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')


##### Reformat data for within-time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  

########## Task

# Get Behavioral Performance

task_performance = get_behavioral_performance(data_2_sample)

# intraclass & corr of % correct
task_reliability = task_rel(task_performance)

########### Comp. Modeling
#get results from Apps&Tzakiris 2013
at_model = model_selection_AT()

#For intepretation see https://www.nicebread.de/a-short-taxonomy-of-bayes-factors/

# ##### fit data sample N=3, T1 & T2 SEPERATE // NO LOOCV // No imput

fit_data_sample_T1 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, True)
fit_data_sample_T2 = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, True)

#save results
path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\output'
fit_data_sample_T1['subject_level_model_evidence'].to_csv(path + '/fit_data_sample_T1.csv')
fit_data_sample_T2['subject_level_model_evidence'].to_csv(path + '/fit_data_sample_T2.csv')

# ##### fit data sample N=3, T1 & T2 SEPERATE // LOOCV // No imput
#TODO
#Check for bugs 

fit_data_CV_df_A = fit_data_CV(data_dict_t1, 0.01, False)
fit_data_CV_df_B = fit_data_CV(data_dict_t2, 0.01, False)

# ##### fit data sample N=3, T1 & T2 COMBINED // NO LOOCV // No imput

fit_data_t1_t2 = data_fit_t1_t2_comb(data_dict_t1,data_dict_t2, 0.01)

# ########### Between time model fitting
# Model Select (t-test procedure, BIC dBIC)
res_tt_func = orig_procedure(fit_data_sample_T1, fit_data_sample_T2)

# Model Select (BIC )
from class_import import reformat_data_within_T,bic_modelselect
res_bic1 = bic_modelselect(fit_data_sample_T1, fit_data_sample_T2)
bic_model_n = [i for i in res_bic1[0]]

from scipy.special import logsumexp
dat_bivc = res_bic1[0].loc[[i for i in res_bic1[0].index if 'A' in i]].sum()

mult = (dat_bivc[bic_model_n[0]] - logsumexp([dat_bivc[i].sum() for i in bic_model_n]))


res_bic_A = res_bic1.T[[i for i in res_bic1.index if 'A' in i]].sum(axis = 1)
res_bic_B = res_bic1.T[[i for i in res_bic1.index if 'B' in i]].sum(axis = 1)

# # Corrected LR ratio
# corr_lr_1 = corr_lr_func(fit_data_sample_T1)
# corr_lr_2 = corr_lr_func(fit_data_sample_T2)

# # Total_corr LR
# time_ag_lr = pd.DataFrame()
# models = [i for i in fit_data_sample_T1['group_level_model_evidence'].index]

# for model in models:
#     res_comb = fit_data_t1_t2.copy().sum(axis = 1)
#     res_A = fit_data_sample_T1['group_level_model_evidence'][model]
#     res_B = fit_data_sample_T2['group_level_model_evidence'][model]
#     lr_agn = (res_A + res_B)/res_comb
#     time_ag_lr[model] = [lr_agn]
# #TODO
# #Really implement?

# ########### Within time model fitting
# #Bayes Between Cond RFX Model Select

# non_ex_p = bayes_RFX_cond(fit_data_sample_T1,fit_data_sample_T2)
# #print('non_exceedence_prob=',float(non_ex_p[1]))1

# # TODO
# # Time-agnostic LR -> Function


# model_names = [i for i in fit_data_sample_T1['group_level_model_evidence'].index]
# subjects = [i[0] for i in fit_data_sample_T1['subject_level_model_evidence']]

# time_agn = pd.DataFrame(index = ['time_agn','within_time'])
# for model in model_names:
#     mod_ev_combined = np.sum(fit_data_t1_t2.copy().T[model])
#     mod_ev_A = np.sum(fit_data_sample_T1['subject_level_model_evidence'].copy().T[model])
#     mod_ev_B = np.sum(fit_data_sample_T2['subject_level_model_evidence'].copy().T[model])
#     per_subj = []
#     per_subj1 = []
#     lr_agn = (mod_ev_A + mod_ev_B) - mod_ev_combined
#     lr_base = (mod_ev_A - mod_ev_B)
#     time_agn[model] = [lr_agn,lr_base]

########### works kinda

# ##### fit data sample N=3, T1 & T2 COMBINED // NO LOOCV // No imput
# data_t1_t2_CV = {**data_dict_t1, **data_dict_t2}
# ids_t1_t2_joblib = [i for i in data_t1_t2_CV.keys()]
# # suppl dat
# # from joblib.externals.loky import set_loky_pickler
# # set_loky_pickler('pickle')
# data_job = []

# for ids in ids_t1_t2_joblib:
#     data = data_t1_t2_CV[ids]
#     data_np = data_t1_t2_CV[ids].to_numpy()
#     data_descr = [i for i in data_t1_t2_CV[ids]]
#     #(unique_id, data, lbfgs_epsilon, verbose_tot)
#     data_job.append([ids,data_np,0.01, False])


# data_np1 = data_np[:,0]
# data_np_hold = data_np[5,:]
# data_np2 = np.delete(data_np1,0,axis=0)
# len(data_np2)    
# if __name__ == '__main__':    
#     res = Parallel(n_jobs=8,verbose=50)(delayed(fit_data_CV_mult)(i) for i in data_job)    








# fit_data_CV_comb = 


########## Data_Pilot sample
# from class_import import data_fit_t1_t2_comb
# fit_data_t1_t2 = data_fit_t1_t2_comb(data_dict_t1,data_dict_t2, 0.01)
# # fit data sample N=10
# fit_data_sample_1 = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)







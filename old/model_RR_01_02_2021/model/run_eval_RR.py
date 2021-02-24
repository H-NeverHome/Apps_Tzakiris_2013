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


########### Retrieve & Preprocess Data

folder_path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\processed'

data_2_sample = get_data_2(r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')


##### Reformat data
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  

########## Face Learning Task

# Get Behavioral Performance

task_performance = get_behavioral_performance(data_2_sample)

# intraclass & corr of % correct
task_reliability = task_rel(task_performance)

########### Comp. Modeling
#get results from Apps&Tzakiris 2013
at_model = model_selection_AT()

#For intepretation see https://www.nicebread.de/a-short-taxonomy-of-bayes-factors/

# ##### fit data, T1 & T2 SEPERATE // NO LOOCV // No imput

fit_data_sample_T1 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, True)
fit_data_sample_T2 = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, True)


# ##### fit data sample N=3, T1 & T2 COMBINED // NO LOOCV // No imput

fit_data_t1_t2 = data_fit_t1_t2_comb(data_dict_t1,data_dict_t2, 0.01)


########### Between time model fitting

# Model Select (t-test procedure, BIC dBIC)
res_tt_func = orig_procedure(fit_data_sample_T1[0], fit_data_sample_T2[0])

# Bayesian Information Criteria (BIC)

res_bic1 = bic_modelselect(fit_data_sample_T1[0], fit_data_sample_T2[0])

bic_subj = res_bic1['subject_wise_BIC']

res_bic_A = bic_subj.T[[i for i in bic_subj.index if 'A' in i]].sum(axis = 1)
res_bic_B = bic_subj.T[[i for i in bic_subj.index if 'B' in i]].sum(axis = 1)

# Corrected LR ratio
corr_lr_1 = corr_lr_func(fit_data_sample_T1[0])
corr_lr_2 = corr_lr_func(fit_data_sample_T2[0])


########### Within time model fitting
#Bayes Between Cond RFX Model Select

non_ex_p = bayes_RFX_cond(fit_data_sample_T1[0],fit_data_sample_T2[0])

# Time agnostic LR
time_ag_lr = pd.DataFrame()
models = [i for i in fit_data_sample_T1[0]['group_level_model_evidence'].index]

for model in models:
    res_comb = fit_data_t1_t2.copy().sum(axis = 1)[model]
    #print('res_comb',res_comb)
    res_A = fit_data_sample_T1[0]['group_level_model_evidence'][model]
    #print('res_A',res_A)
    res_B = fit_data_sample_T2[0]['group_level_model_evidence'][model]
    #print('res_B',res_B)
    lr_agn = res_comb-(res_A + res_B)
    #print('lr_agn',lr_agn)
    time_ag_lr[model] = [lr_agn]
    
    

########### Save Data
path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\output_RR'

# get time

datetime = str(pd.to_datetime("now"))
date,time = datetime.split(' ')
datetime_fin = date+'_'+ 'h'+time[0:5].replace(":", "m")
print(datetime_fin)

# create new folder
new_folder = path+'/'+datetime_fin
os.mkdir(path+'/'+datetime_fin)
path_new = new_folder
### save BIC 
BIC_res = pd.DataFrame(index = [i for i in res_bic_A.index])
BIC_res['T_1'] = res_bic_A 
BIC_res['T_2'] = res_bic_B

BIC_res.to_csv(path_new + '/BIC_res_' + datetime_fin + '.csv')

### save raw LL
raw_LL_T1 = fit_data_sample_T1[0]['group_level_model_evidence']
raw_LL_T2 = fit_data_sample_T2[0]['group_level_model_evidence']
LL_res = pd.DataFrame(index = [i for i in raw_LL_T1.index])
LL_res['T_1'] = raw_LL_T1
LL_res['T_2'] = raw_LL_T2

LL_res.to_csv(path_new + '/LL_res_' + datetime_fin + '.csv')

### Corrected LR ratio
corr_LR_T1 = corr_lr_1['corr_lr'].T 
corr_LR_T2 = corr_lr_2['corr_lr'].T 

corr_LR = pd.DataFrame(index=[i for i in corr_LR_T1.index])
corr_LR['T_1'] = corr_LR_T1['corr_lr']
corr_LR['T_2'] = corr_LR_T2['corr_lr']

corr_LR.to_csv(path_new + '/corr_LR_' + datetime_fin + '.csv')
### Time agnostic LR ratio & rfx
time_ag_lr['non_ex_prob'] = non_ex_p[1]
time_ag_lr.to_csv(path_new + '/time_ag_lr_' + datetime_fin + '.csv')







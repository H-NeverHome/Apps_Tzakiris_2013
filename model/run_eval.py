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
from class_import import reformat_data_within_T,bic,fit_data_CV
import matlab.engine
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


##### Reformat data for within time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  

########## Task

# Get Behavioral Performance
task_performance = get_behavioral_performance(data_2_sample)

# intraclass corr
icc_data = task_performance['behavioral_results'].copy()
icc_data['time'] = [i[-1] for i in icc_data.index]


########### Comp. Modeling
#get results from Apps&Tzakiris 2013
at_model = model_selection_AT()

#For intepretation see https://www.nicebread.de/a-short-taxonomy-of-bayes-factors/


# ##### fit data sample N=3, T1 & T2 SEPERATE // NO LOOCV // No imput

fit_data_sample_T1 = fit_data_noCV_irr_len_data(data_dict_t1, 0.01, False)
fit_data_sample_T2 = fit_data_noCV_irr_len_data(data_dict_t2, 0.01, False)

# ########### Between time model fitting
# Model Select (t-test procedure, BIC dBIC)
res_tt_func = orig_procedure(fit_data_sample_T1, fit_data_sample_T2)

# Model Select (BIC )
res_bic1 = bic(fit_data_sample_T1, fit_data_sample_T2)

# Corrected LR ratio


# ########### Within time model fitting
# Bayes Between Cond RFX Model Select

# non_ex_p = bayes_RFX_cond(fit_data_sample_T1,fit_data_sample_T2)
# print('non_exceedence_prob=',float(non_ex_p[1]))

# ########### TODO

# # ##### fit data sample N=3, T1 & T2 COMBINED // NO LOOCV // No imput
# #fit_data_t1_t2 = data_fit_t1_t2_comb(data_dict_t1,data_dict_t2, 0.01)

#fit_data_CV_df = fit_data_CV(data_dict_t1, 0.01, False)
#fit_data_CV_df = fit_data_CV(data_dict_t2, 0.01, False)



########## Data_Pilot sample

# # fit data sample N=10
# fit_data_sample_1 = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)







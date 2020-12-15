# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:43:15 2020

@author: de_hauk
"""
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'C:\Program Files\Polyspace\tapas-master\HGF')
import numpy as np
import pandas as pd
import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import reformat_data_within_T,bic,fit_data_CV

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')


##### Reformat data for within-time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  


# HGF


inputs = np.array(data_dict_t1['1_A']['stim_IDs']).tolist()
aaa = np.zeros([len(inputs)])
aaa[::] = inputs
inputs_fin = matlab.double(aaa.tolist())


outputs = np.array(data_dict_t1['1_A']['1_A_answer']).tolist()
bbb = np.zeros([len(outputs)])
bbb[::] = outputs
outputs_fin = matlab.double(bbb.tolist())

est = eng.tapas_fitModel(outputs_fin, inputs_fin)

res = pd.DataFrame()
est_d = list(est['p_prc'])
for i in est_d:
    dat = est['p_prc'][i]
    res[i] = np.asarray(dat)

# est = eng.tapas_fitModel([],inputs,
#                          'tapas_hgf_binary_config',
#                          'tapas_bayes_optimal_binary_config',
#                          'tapas_quasinewton_optim_config')

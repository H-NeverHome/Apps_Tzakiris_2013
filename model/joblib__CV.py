# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:56:42 2020

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
data_t1_t2_CV = {**data_dict_t1, **data_dict_t2}

ids_t1_t2_joblib = [i for i in data_t1_t2_CV.keys()]



# suppl dat
# from joblib.externals.loky import set_loky_pickler
# set_loky_pickler('pickle')
data_job = []

for ids in ids_t1_t2_joblib:
    data = data_t1_t2_CV[ids]
    data_np = data_t1_t2_CV[ids].to_numpy()
    data_descr = [i for i in data_t1_t2_CV[ids]]
    #(unique_id, data, lbfgs_epsilon, verbose_tot)
    data_job.append([ids,data_np,0.01, False])


data_np1 = data_np[:,0]
data_np_hold = data_np[5,:]
data_np2 = np.delete(data_np1,0,axis=0)
len(data_np2)    
if __name__ == '__main__':    
    res = Parallel(n_jobs=8,verbose=50)(delayed(fit_data_CV_mult)(i) for i in data_job)    

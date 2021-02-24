# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:56:42 2020

@author: de_hauk
"""



import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data_2
from class_import import reformat_data_within_T
from class_import import fit_data_CV_mult
from joblib import Parallel, delayed
import pandas as pd

########### Get Data

folder_path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\processed'
#data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

##### Reformat data for within-time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]

### Here into function 

# def fit_data_CV_multp(data_dict_t1,data_dict_t2) 

### merge dataframes
data_t1_t2_CV = {**data_dict_t1, **data_dict_t2}

### multiprocessing -> each ID alloc one process
ids_t1_t2_joblib = [i for i in data_t1_t2_CV.keys()]



### data reformating into np.arrays
data_job = []

for ids in ids_t1_t2_joblib:
    data = data_t1_t2_CV[ids]
    data_np = data_t1_t2_CV[ids].to_numpy()
    data_np[:,2] = data_np[:,2].astype(int)
    data_descr = [i for i in data_t1_t2_CV[ids]]
    #(unique_id, data, lbfgs_epsilon, verbose_tot)
    data_job.append([ids, data_np, 0.01, False])

# from sklearn.model_selection import LeaveOneOut
# #np.delete(curr_data_vpn.copy(),indx,axis=0)
# loocv = LeaveOneOut()

# data_cv = [i for i in loocv.split(x=data_job[0][1], y=data_job[0][1][:,2]) ]


# import numpy as np
# indx = 9  

# curr_data_vpn = data_job[0][1]
    
# # holdout-data
# holdout_data = curr_data_vpn.copy()[indx,:]
# action = curr_data_vpn.copy()[indx,:][2]

# # training data
# train_data = np.delete(curr_data_vpn.copy(),indx,axis=0)



if __name__ == '__main__':   
    results123 = Parallel(n_jobs=8,verbose=50)(delayed(fit_data_CV_mult)(i) for i in data_job)    

resCV_A = [i[[i for i in i.keys()][0]] for i in results123[0:3]]
resCV_B = [i[[i for i in i.keys()][0]] for i in results123[3::]]

indx = resCV_A[0].index
fin_res_A = pd.DataFrame(index = indx)
fin_res_B = pd.DataFrame(index = indx)
for ids,data_res in zip(ids_t1_t2_joblib, results123):
    curr_dat = data_res[ids][0]
    if 'A' in ids:
        fin_res_A[ids] = curr_dat
    elif 'B' in ids:
        fin_res_B[ids] = curr_dat

path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\output'
fin_res_A.to_csv(path + '/fit_data_sample_T1_CV.csv')
fin_res_B.to_csv(path + '/fit_data_sample_T2_CV.csv')




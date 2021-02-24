# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:25:34 2021

@author: de_hauk
"""

import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data_2
from class_import import reformat_data_within_T
from class_import import fit_data_CV_mult
from joblib import Parallel, delayed
import pandas as pd

from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT_CV,VIEW_DEPENDENT_CV
from model_functions_BFGS import VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV,VIEW_INDEPENDENT_CV
from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENT_CV
from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV
import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize
np.random.seed(1993)


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




# #######

# for every vpn_data:
#     for every trial in vpn_data:
#         split_data
#         opt training-data

# ######






unique_id, data, lbfgs_epsilon, verbose_tot = data_job[0]
vpn = unique_id

epsilon_param = lbfgs_epsilon
x_0_bfgs = 0.5

total_results = {}



curr_data_vpn = data

cv_score_view_ind_cont = []
cv_score_view_dep = []
cv_score_view_dep_cont=[]
cv_score_view_ind = []
cv_score_view_ind_dep = []
cv_score_view_ind_dep_cont = []
cv_score_rnd = []

# for trial in trial
''' one trial needs to be subtracted from the data, 
since we are deleting one trial'''

trials_n = len(curr_data_vpn)


models = []
#### debug
test_L = []
hold_L = []  

for indx in range(trials_n):
    print(indx)
    
    # Data Index -> [view-Dep-StimID, correct?, answer, V_ind StimID, n_prev_pres]
    
    # holdout-data
    holdout_data = curr_data_vpn.copy()[indx,:]
    action = curr_data_vpn.copy()[indx,:][2]
    
    # training data
    train_data = np.delete(curr_data_vpn.copy(),indx,axis=0)
    
    # delete one index -> holdout-data
    curr_index = indx-1
    
    # data import
    stim_IDs                = train_data[:,3].copy()   #stimulus IDs of winning model 
    new_ID                  = train_data[:,4].copy()   #trials where new ID is introduced 
    numb_prev_presentations = train_data[:,5].copy()   #number_of_prev_presentations
    stim_IDs_perspective    = train_data[:,0].copy()   #view dependent
    VPN_output              = train_data[:,2].copy()   #VPN answers
    verbose                 = False

    ##### Model Optim
    #data_raw.append()
    test_L.append(holdout_data)
    hold_L.append(train_data)
########################################## VIEW_INDIPENDENTxCONTEXT #####################################             

    data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

    bounds_M1 = [(.0,1.),(0.,1.),(1.,20.),(0.,2.)]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
    res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                  approx_grad = True,
                                  bounds = bounds_M1, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                  epsilon=epsilon_param)   

    data_M1[-1] = True 
    data_M1_debug = data_M1.copy()
    params_m_1 = res1[0]
    m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
    try:
        init_V_m_1 = m_1[1]['history_V_dict'][holdout_data[3]][holdout_data[5]-1]
    except:
        init_V_m_1 = m_1[1]['history_V_dict'][holdout_data[3]][-1]
    init_C_m_1 = np.array(m_1[1]['data_store_1']['history_C'])[curr_index]
    

    cv_trial_indeXcontext,totfam = VIEW_INDIPENDENTxCONTEXT_CV(params_m_1,
                                                               init_V_m_1,
                                                               init_C_m_1,
                                                               action)
    
    
###################################### VIEW_DEPENDENT ######################################            


    data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
    
    data_M2_debug = data_M2.copy()
    data_M2_debug[-1] = True 
    bounds_M2 = [(0,1),(.1,20),(0,2)]
    
    part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
    res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                  approx_grad = True,
                                  bounds = bounds_M2, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                  epsilon=epsilon_param)
    params_m_2 = res2[0]
    #parameter_est['VIEW_DEPENDENT'][vpn] = res2[0]

    
    m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
    init_V_m_2 = 0
    
    v_dep_dict = m_2[1]['history_V']
    try:
        init_V_m_2 = v_dep_dict[holdout_data[0]][holdout_data[5]-1]
    except:
        try:
            init_V_m_2 = v_dep_dict[holdout_data[0]][-1]
        except:
            init_V_m_2 = m_2[1]['init_val']


    cv_trial_dep = VIEW_DEPENDENT_CV(params_m_2,
                                      init_V_m_2,
                                      action)
    
######################################### VIEW_DEPENDENTxCONTEXT_DEPENDENT ##################################### 
            
    data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
    bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]

    part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
    res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                  approx_grad = True,
                                  bounds = bounds_M3, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                  epsilon=epsilon_param)

    data_M3_debug = data_M3.copy()
    data_M3_debug[-1] = True 
    params_m_3 = res3[0]
    m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
    
    v_dep_dict_M3 = m_3[1]['history_V_dict']
    try:
        init_V_m_3 = v_dep_dict_M3[holdout_data[0]][holdout_data[5]-1]
    except:
        try:
            init_V_m_3 = v_dep_dict_M3[holdout_data[0]][-1]
        except:
            init_V_m_3 = m_3[1]['VD_init']

    init_C_m_3 = np.array(m_3[1]['history_C'])[curr_index]

    cv_trial_dep_cont = VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV(params_m_3,
                                                            init_V_m_3,
                                                            init_C_m_3,
                                                            action)    
########################### rnd_choice ###################################################     
    '''for every answer predicted model prob of =.5'''
    ans_prob_rnd = np.log(.5)


##############################################################################        
    
    cv_score_view_ind_cont.append(cv_trial_indeXcontext)
    cv_score_view_dep.append(cv_trial_dep)
    cv_score_view_dep_cont.append(cv_trial_dep_cont)
    cv_score_rnd.append(ans_prob_rnd)
    models.append({'VIEW_INDIPENDENTxCONTEXT': {'model_res':m_1,
                                                'cv_score':cv_trial_indeXcontext,
                                                'params':params_m_1,
                                                'V':init_V_m_1,
                                                'C':init_C_m_1,
                                                'totfam':totfam},
                   'VIEW_DEPENDENT': [m_2,cv_trial_dep],
                   'action': action,
                   'holdout': holdout_data,
                   'train': train_data})
    
    
df_index = ['VIEW_INDIPENDENTxCONTEXT',
            'VIEW_DEPENDENT',
            'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
            'RANDOM_CHOICE']

df_data = [np.array(cv_score_view_ind_cont).sum(),
           np.array(cv_score_view_dep).sum(),
           np.array(cv_score_view_dep_cont).sum(),
           np.array(cv_score_rnd).sum()]

cv_trial = pd.DataFrame(data = df_data, index=df_index)
 
total_results[vpn] = cv_trial
total_results['suppl'] = (test_L,hold_L)
total_results['error'] = (cv_score_view_ind_cont,df_data,m_1)
total_results['VPN'] = curr_data_vpn    


    
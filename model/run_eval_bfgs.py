# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:25:42 2020

@author: de_hauk
"""


from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'



import pandas as pd
import numpy as np
import datetime
from functools import partial
import scipy
from scipy import optimize
np.random.seed(1993)
### Get data 
data = pd.read_csv(folder_path_data + r'/final_proc_dat_labjansen.csv')

#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]

epsilon_param = 0.01
x_0_bfgs = 0.7




models_names = ['VIEW_INDIPENDENTxCONTEXT',
                'VIEW_DEPENDENT',
                'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                'VIEW_INDEPENDENT',
                'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
                
results_history_optim = {'VIEW_INDIPENDENTxCONTEXT': [],
                         'VIEW_DEPENDENT': [],
                         'VIEW_DEPENDENTxCONTEXT_DEPENDENT': [],
                         'VIEW_INDEPENDENT': [],
                         'VIEW_INDEPENDENTxVIEW_DEPENDENT':[],
                         'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT':[],
                         'IDs':sample_answer_clms,
                         'models':models_names}

res_evidence = pd.DataFrame(index=models_names)


for i,j in zip(sample_answer_clms,sample_perspective_clms):
    print(i)
    #func calls & rand starts

    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data['number_of_prev_presentations_raw '] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    verbose = False
    
    
    ##### Model Optim
    
    
    print('VIEW_INDIPENDENTxCONTEXT')
    data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    #[alpha, sigma, beta, lamd_a,]
    bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
    res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                  approx_grad = True,
                                  bounds = bounds_M1, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                  epsilon=epsilon_param)
    
    
    
    print('VIEW_DEPENDENT')
    #params = [alpha, beta, lamd_a]
    #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
    
    data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]


    bounds_M2 = [(0,1),(.1,20),(0,2)]
    
    part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
    res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                  approx_grad = True,
                                  bounds = bounds_M2, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                  epsilon=epsilon_param)
    
    
    print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')
    #params = [alpha, sigma, beta, lamd_a,]
    #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
    
    data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
    bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]

    part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
    res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                  approx_grad = True,
                                  bounds = bounds_M3, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                  epsilon=epsilon_param)
    
    print('VIEW_INDEPENDENT')
    #params = [alpha, beta, lamd_a]
    #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

    data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    bounds_M4 = [(0,1),(.1,20),(0,2)]

    part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
    res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                  approx_grad = True,
                                  bounds = bounds_M4, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                  epsilon=epsilon_param)
    
    
    print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
    #params = [alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep]
    #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]


    data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
    res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                  approx_grad = True,
                                  bounds = bounds_M5, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                  epsilon=epsilon_param)
    
    print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
    #params = [alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep]
    #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]

    data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
    res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                  approx_grad = True,
                                  bounds = bounds_M6, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                  epsilon=epsilon_param)

    res_evidence[i] = [i[1] for i in [res1,res2,res3,res4,res5,res6]]
    
restotal = res_evidence.sum(axis=1)

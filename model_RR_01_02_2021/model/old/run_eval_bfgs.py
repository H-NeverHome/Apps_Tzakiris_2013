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
from functools import partial
import scipy
from scipy import optimize
from scipy import special
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
                
data_verbose_debug = {}
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
    ############################## Verbose == True ###########################
    
    verbose_debug = True
    if verbose_debug == True:
        #data_M1_debug = data_M1
        data_M_debug = [data_M1, data_M2, data_M3, data_M4, data_M5, data_M6]
        for dat in data_M_debug:
            dat[-1] = True
    
        
        
        data_M1_debug = data_M_debug[0]
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
        
        data_M2_debug = data_M_debug[1]
        params_m_2 = res2[0]
        m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
    
        data_M3_debug = data_M_debug[2]
        params_m_3 = res3[0]
        m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
        
        data_M4_debug = data_M_debug[3]
        params_m_4 = res4[0]
        m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
    
        data_M5_debug = data_M_debug[4]
        params_m_5 = res5[0]
        m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
        
        data_M6_debug = data_M_debug[5]
        params_m_6 = res6[0]
        m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)

        res_debug = {models_names[0]: m_1,
                     models_names[1]: m_2,
                     models_names[2]: m_3,
                     models_names[3]: m_4,
                     models_names[4]: m_5,
                     models_names[5]: m_6}
        data_verbose_debug[i] = res_debug
restotal = res_evidence.sum(axis=1)
cntrl_log = special.logsumexp(-1*np.array(restotal[1::]))
bf_log = (-1*np.array(restotal[0])) - cntrl_log

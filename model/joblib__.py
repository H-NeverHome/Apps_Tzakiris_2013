# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:31:40 2020

@author: de_hauk
"""

# import os
# os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
# from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
# from class_import import get_behavioral_performance,model_selection_AT
# from class_import import fit_data_noCV_irr_len_data,data_fit_t1_t2_comb
# #from class_import import fit_data_NUTS
# from class_import import bayes_RFX_cond,orig_procedure
# from class_import import reformat_data_within_T,bic,fit_data_CV
# from class_import import task_rel,corr_lr_func
# import matlab.engine
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import pingouin as pg
# from scipy import special 


from math import sqrt
from joblib import Parallel, delayed
import time
import numpy as np

from scipy.optimize import rosen,fmin_bfgs
rnd_state = np.random.RandomState(seed=1993)
data123 = [rnd_state.randint(1, 100, 100) for i in range(10)]


def rose(init_point):
    res1 = fmin_bfgs(rosen,
                     x0 = init_point,
                     epsilon=0.000001,
                     full_output = True)
    return res1

start_time = time.time()
res_1234 = []   
for i in data123:
    res123 = rose(i)
    res_1234.append(res123)

duration_nomulti =(time.time() - start_time)
start_time2 = time.time()
if __name__ == '__main__':    
    res = Parallel(n_jobs=16)(delayed(rose)(i) for i in data123)
duration_multi =(time.time() - start_time2)




# 
# iters = 100000

# ########### Get Data

# folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
# #data_1_sample = get_data(folder_path_data)

# data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
#                       r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')


# from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
# from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
# from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT

# #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'

# ##### Reformat data for within-time format
# data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
# data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  


# import pandas as pd
# import numpy as np
# from functools import partial
# from scipy import optimize,special
# np.random.seed(1993)
# ### Get data 
# data = data_dict_t1

# #### get unique IDS
# unique_id = list(data.keys())
# sample_answer_clms = [i+'_answer' for i in unique_id]
# sample_perspective_clms = [i+'_perspective' for i in unique_id]

# epsilon_param = .01
# x_0_bfgs = 0.5
# params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
# params_M2_name = ['alpha', 'beta', 'lamd_a']
# params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
# params_M4_name = ['alpha', 'beta', 'lamd_a']
# params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
# params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

# models_names = ['VIEW_INDIPENDENTxCONTEXT',
#                 'VIEW_DEPENDENT',
#                 'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
#                 'VIEW_INDEPENDENT',
#                 'VIEW_INDEPENDENTxVIEW_DEPENDENT',
#                 'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
#                 'random_choice']

# parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
#                 'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
#                 'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
#                 'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
#                 'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
#                 'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
#                     }
                
# data_verbose_debug = {}
# res_evidence = pd.DataFrame(index=models_names)
# trialwise_data = {}
# bf_log_group = pd.DataFrame()

# for vpn in unique_id:
#     print(vpn)
#     #func calls & rand starts
    
#     curr_data_vpn = data[vpn]
#     # data import
#     stim_IDs = curr_data_vpn['stim_IDs'] #stimulus IDs of winning model 
#     new_ID = curr_data_vpn['new_IDs'] #trials where new ID is introduced 
#     numb_prev_presentations = curr_data_vpn['n_prev_pres'] #number_of_prev_presentations
#     stim_IDs_perspective = curr_data_vpn[vpn+'_perspective'] #view dependent
#     VPN_output = curr_data_vpn[vpn+'_answer'] #VPN answers
#     verbose = False
    
    
#     ##### Model Optim
    
#     i=vpn
#     print('VIEW_INDIPENDENTxCONTEXT')
#     data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

#     bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
#     part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
#     res1 = optimize.fmin_l_bfgs_b(part_func_M1,
#                                   approx_grad = True,
#                                   bounds = bounds_M1, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M1))],
#                                   epsilon=epsilon_param)
#     parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
    
    
#     print('VIEW_DEPENDENT')
    
#     #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
#     data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]


#     bounds_M2 = [(0,1),(.1,20),(0,2)]
    
#     part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
#     res2 = optimize.fmin_l_bfgs_b(part_func_M2,
#                                   approx_grad = True,
#                                   bounds = bounds_M2, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M2))],
#                                   epsilon=epsilon_param)
    
#     parameter_est['VIEW_DEPENDENT'][i] = res2[0]
    
    
#     print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

#     #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
    
#     data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
#     bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]

#     part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
#     res3 = optimize.fmin_l_bfgs_b(part_func_M3,
#                                   approx_grad = True,
#                                   bounds = bounds_M3, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M3))],
#                                   epsilon=epsilon_param)
#     parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
    
#     print('VIEW_INDEPENDENT')

#     #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

#     data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
#     bounds_M4 = [(0,1),(.1,20),(0,2)]

#     part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
#     res4 = optimize.fmin_l_bfgs_b(part_func_M4,
#                                   approx_grad = True,
#                                   bounds = bounds_M4, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M4))],
#                                   epsilon=epsilon_param)
#     parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
    
    
#     print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
    
#     #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]


#     data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
#     bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]

#     part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
#     res5 = optimize.fmin_l_bfgs_b(part_func_M5,
#                                   approx_grad = True,
#                                   bounds = bounds_M5, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M5))],
#                                   epsilon=epsilon_param)
#     parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
    

#     print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
#     params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
#     #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]

#     data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
#     bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]

#     part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
#     res6 = optimize.fmin_l_bfgs_b(part_func_M6,
#                                   approx_grad = True,
#                                   bounds = bounds_M6, 
#                                   x0 = [x_0_bfgs for i in range(len(bounds_M6))],
#                                   epsilon=epsilon_param)
    
#     parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
    
    
#     ##### RND Choice model
#     rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
    
#     re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
#     res_evidence[i] = re_evidence_subj
    
#     ### Subject BF_LOG
#     bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
#     bf_log_group[i + '_BF_log'] = [bf_log_subj]

    
#     ############################## Verbose == True ###########################
    
#     verbose_debug = verbose_tot
    
#     data_M_debug = [data_M1, data_M2, data_M3, data_M4, data_M5, data_M6]
#     for dat in data_M_debug:
#         dat[-1] = True        
    
#     if verbose_debug == True:

#         data_M1_debug = data_M_debug[0]
#         params_m_1 = res1[0]
#         m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
        
#         data_M2_debug = data_M_debug[1]
#         params_m_2 = res2[0]
#         m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
    
#         data_M3_debug = data_M_debug[2]
#         params_m_3 = res3[0]
#         m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
        
#         data_M4_debug = data_M_debug[3]
#         params_m_4 = res4[0]
#         m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
    
#         data_M5_debug = data_M_debug[4]
#         params_m_5 = res5[0]
#         m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
        
#         data_M6_debug = data_M_debug[5]
#         params_m_6 = res6[0]
#         m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)

#         res_debug = {models_names[0]: m_1,
#                      models_names[1]: m_2,
#                      models_names[2]: m_3,
#                      models_names[3]: m_4,
#                      models_names[4]: m_5,
#                      models_names[5]: m_6}
#         data_verbose_debug[i] = res_debug
        
# #### Get winning model trialwise dat ####
#     data_M1_debug = data_M_debug[0]
#     params_m_1 = res1[0]
#     m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)        

#     trialwise_data[i] = m_1[1]['data_store_1']
# restotal = res_evidence.sum(axis=1)
# cntrl_log = special.logsumexp(np.array(restotal[1::]))
# bf_log = (cntrl_log -(np.array(restotal[0])))

# results_1 = {'uncorr_LR_10':np.exp(-1*bf_log),
#             'subject_level_model_evidence':res_evidence,
#             'group_level_model_evidence':res_evidence.sum(axis=1),
#             'subject_level_uncorr_LR': bf_log_group,
#             'xxx':data_verbose_debug,
#             'used_data': data,
#             'subject_level_parameter-estimates':parameter_est,
#             'subject_level_trialwise_data_win_model':trialwise_data,
#             'baseline_model': np.sum(np.log([0.5 for i in range(189)]))}
# if verbose_tot==True:
#     return (results_1,restotal,data_verbose_debug)
# elif verbose_tot==False:
#     return results_1
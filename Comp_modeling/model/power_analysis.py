# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:16:24 2021

@author: de_hauk
"""
import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize,special
np.random.seed(1993)
import numpy as np
import pandas as pd
import glob
import random 
from joblib import Parallel, delayed
import os
pathtomodels = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'

os.chdir(pathtomodels)

from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic



def bounds(x):
    if x<=.0:
        return .0001
    elif x>=1.:
        return .9999
    else:
        return x

def fit_data_seperate(path_to_modelfunctions, data, verbose_tot):
       import os
       os.chdir(path_to_modelfunctions)
       
       from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
       from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
       from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

       import pandas as pd
       import numpy as np
       from functools import partial
       from scipy import optimize,special
       np.random.seed(1993)

       #### get unique IDS
       unique_id = list(data.keys())

       
       epsilon_param = .01
       x_0_bfgs = 0.5
       params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
       params_M2_name = ['alpha', 'beta', 'lamd_a']
       params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
       params_M4_name = ['alpha', 'beta', 'lamd_a']
       params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
       params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
   
       models_names = ['VIEW_INDIPENDENTxCONTEXT',
                       'VIEW_DEPENDENT',
                       'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                       'VIEW_INDEPENDENT',
                       'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                       'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
                       'random_choice']
       models_names_bic = ['VIEW_INDIPENDENTxCONTEXT',
                           'VIEW_DEPENDENT',
                           'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                           'VIEW_INDEPENDENT',
                           'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                           'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
       
       parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                       'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                       'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                       'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                       'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                       'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                           }
                       
       data_verbose_debug = {}
       res_evidence = pd.DataFrame(index=models_names)
       trialwise_data = {}
       bf_log_group = pd.DataFrame()
       
       #data_mult =[data[vpn] for vpn in unique_id]
       
       for vpn in unique_id:
           #func calls & rand starts
           print(vpn)

           curr_data_vpn = data[vpn]

           # get data
           stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
           new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
           n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
           stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
           VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
           verbose =       False
           
           # # get data numpy
           # stim_IDs_VI=    curr_data_vpn[2]  #stimulus IDs of winning model 
           # new_ID=         curr_data_vpn[1]       #trials where new ID is introduced 
           # n_prev_pres=    curr_data_vpn[4]    #number_of_prev_presentations
           # stim_IDs_VD=    curr_data_vpn[3]  #view dependent
           # VPN_output =    curr_data_vpn[0]     #VPN answers
           # verbose =       False
           #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

           data_ALL = [VPN_output.astype(int), 
                       new_ID.astype(int), 
                       n_prev_pres.astype(int), 
                       stim_IDs_VI.astype(str), 
                       stim_IDs_VD.astype(str), 
                       verbose]            

           
           
           ########## Model Optim
           
           i=vpn
           print('VIEW_INDIPENDENTxCONTEXT')
   
           bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
           
           part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
           res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                         approx_grad = True,
                                         bounds = bounds_M1, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
           
           ##### VIEW_DEPENDENT
           print('VIEW_DEPENDENT')
                   
           bounds_M2 = [(0,1),(.1,20),(0,2)]
           
           part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
           res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                         approx_grad = True,
                                         bounds = bounds_M2, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                         epsilon=epsilon_param)
           
           parameter_est['VIEW_DEPENDENT'][i] = res2[0]
           
           ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
           print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

           bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
       
           part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
           res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                         approx_grad = True,
                                         bounds = bounds_M3, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
           
           ##### VIEW_INDEPENDENT
           print('VIEW_INDEPENDENT')
   
           bounds_M4 = [(0,1),(.1,20),(0,2)]
       
           part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
           res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                         approx_grad = True,
                                         bounds = bounds_M4, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
           
           ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
           print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
                   
           bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
       
           part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
           res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                         approx_grad = True,
                                         bounds = bounds_M5, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
           
           ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
           print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
           #params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
       
           bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
       
           part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL,None) 
           res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                         approx_grad = True,
                                         bounds = bounds_M6, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                         epsilon=epsilon_param)
           
           parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
           
           
           ##### RND Choice model
           rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
           
           re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
           res_evidence[i] = re_evidence_subj
           
           ### Subject BF_LOG
           bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
           bf_log_group[i + '_BF_log'] = [bf_log_subj]
   
           
           ############################## Verbose == True ###########################
           
           verbose_debug = verbose_tot
           
           data_ALL_debug = data_ALL
           data_ALL_debug[-1] = True        
           
           if verbose_debug == True:
   
               # = data_M_debug[0]
               params_m_1 = res1[0]
               m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None,params_m_1)
               
               #data_M2_debug = data_M_debug[1]
               params_m_2 = res2[0]
               m_2 = VIEW_DEPENDENT(data_ALL_debug,None, params_m_2)
           
               #data_M3_debug = data_M_debug[2]
               params_m_3 = res3[0]
               m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_ALL_debug,None, params_m_3)
               
               #ädata_M4_debug = data_M_debug[3]
               params_m_4 = res4[0]
               m_4 = VIEW_INDEPENDENT(data_ALL_debug,None, params_m_4)
           
               #data_M5_debug = data_M_debug[4]
               params_m_5 = res5[0]
               m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_ALL_debug,None, params_m_5)
               
               #data_M6_debug = data_M_debug[5]
               params_m_6 = res6[0]
               m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_ALL_debug,None, params_m_6)
       
               res_debug = {models_names[0]: m_1,
                            models_names[1]: m_2,
                            models_names[2]: m_3,
                            models_names[3]: m_4,
                            models_names[4]: m_5,
                            models_names[5]: m_6}
               data_verbose_debug[i] = res_debug
               
       #### Get winning model trialwise dat ####
           params_m_1 = res1[0]
           res_M1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None, params_m_1)        
   
           trialwise_data[i] = res_M1[1]['data_store_1']
       
       ### some calculation
       restotal = res_evidence.sum(axis=1)
       group_level_me = res_evidence[[i for i in res_evidence]].sum(axis=1)

       
       ##### BIC calc
       indxs = ['bic','raw_LL','n_params', 'sample_size']
       bic_fit= pd.DataFrame(index=indxs)
       for model in models_names_bic:
           n_params = len(list(parameter_est[model].index))
           sample_size = len([i for i in res_evidence])
           raw_LL = group_level_me[model]
           bic_curr = bic(n_params, sample_size, raw_LL)
           bic_fit[model] = [bic_curr,raw_LL,n_params, sample_size]
       
       ## summarize data
       results_1 = {'subject_level_model_evidence':res_evidence,
                   'group_level_me': group_level_me,
                   'group_level_BIC':bic_fit,
                   'used_data': data,
                   'subject_level_parameter-estimates':parameter_est,
                   'subject_level_trialwise_data_win_model':trialwise_data}
       #self.fit_separate = results_1
       if verbose_tot==True:
           return (results_1,restotal,data_verbose_debug)
       elif verbose_tot==False:
           return results_1


def gen_data_vixcontext(path_to_modelfunctions, data, verbose_tot):
       import os
       os.chdir(path_to_modelfunctions)
       
       from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
       from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
       from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

       import pandas as pd
       import numpy as np
       from functools import partial
       from scipy import optimize,special
       np.random.seed(1993)

       #### get unique IDS
       unique_id = list(data.keys())

       
       epsilon_param = .01
       x_0_bfgs = 0.5
       params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
       params_M2_name = ['alpha', 'beta', 'lamd_a']
       params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
       params_M4_name = ['alpha', 'beta', 'lamd_a']
       params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
       params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
   
       models_names = ['VIEW_INDIPENDENTxCONTEXT',
                       'VIEW_DEPENDENT',
                       'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                       'VIEW_INDEPENDENT',
                       'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                       'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
                       'random_choice']
       models_names_bic = ['VIEW_INDIPENDENTxCONTEXT',
                           'VIEW_DEPENDENT',
                           'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                           'VIEW_INDEPENDENT',
                           'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                           'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
       
       parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                       'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                       'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                       'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                       'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                       'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                           }
                       
       data_verbose_debug = {}
       res_evidence = pd.DataFrame(index=models_names)
       trialwise_data = {}
       bf_log_group = pd.DataFrame()
       
       #data_mult =[data[vpn] for vpn in unique_id]
       
       for vpn in unique_id:
           #func calls & rand starts
           print(vpn)

           curr_data_vpn = data[vpn]

           # get data
           stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
           new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
           n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
           stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
           VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
           verbose =       False
           
           # # get data numpy
           # stim_IDs_VI=    curr_data_vpn[2]  #stimulus IDs of winning model 
           # new_ID=         curr_data_vpn[1]       #trials where new ID is introduced 
           # n_prev_pres=    curr_data_vpn[4]    #number_of_prev_presentations
           # stim_IDs_VD=    curr_data_vpn[3]  #view dependent
           # VPN_output =    curr_data_vpn[0]     #VPN answers
           # verbose =       False
           #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

           data_ALL = [VPN_output.astype(int), 
                       new_ID.astype(int), 
                       n_prev_pres.astype(int), 
                       stim_IDs_VI.astype(str), 
                       stim_IDs_VD.astype(str), 
                       verbose]            

           
           
           ########## Model Optim
           
           i=vpn
           print('VIEW_INDIPENDENTxCONTEXT')
   
           bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
           
           part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
           res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                         approx_grad = True,
                                         bounds = bounds_M1, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
           
           ##### VIEW_DEPENDENT
           print('VIEW_DEPENDENT')
                   
           bounds_M2 = [(0,1),(.1,20),(0,2)]
           
           part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
           res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                         approx_grad = True,
                                         bounds = bounds_M2, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                         epsilon=epsilon_param)
           
           parameter_est['VIEW_DEPENDENT'][i] = res2[0]
           
           ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
           print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

           bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
       
           part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
           res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                         approx_grad = True,
                                         bounds = bounds_M3, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
           
           ##### VIEW_INDEPENDENT
           print('VIEW_INDEPENDENT')
   
           bounds_M4 = [(0,1),(.1,20),(0,2)]
       
           part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
           res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                         approx_grad = True,
                                         bounds = bounds_M4, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
           
           ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
           print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
                   
           bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
       
           part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
           res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                         approx_grad = True,
                                         bounds = bounds_M5, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                         epsilon=epsilon_param)
           parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
           
           ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
           print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
           #params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
       
           bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
       
           part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL,None) 
           res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                         approx_grad = True,
                                         bounds = bounds_M6, 
                                         x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                         epsilon=epsilon_param)
           
           parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
           
           
           ##### RND Choice model
           rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
           
           re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
           res_evidence[i] = re_evidence_subj
           
           ### Subject BF_LOG
           bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
           bf_log_group[i + '_BF_log'] = [bf_log_subj]
   
           
           ############################## Verbose == True ###########################
           
           verbose_debug = verbose_tot
           
           data_ALL_debug = data_ALL
           data_ALL_debug[-1] = True        
           
           if verbose_debug == True:
   
               # = data_M_debug[0]
               params_m_1 = res1[0]
               m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None,params_m_1)
               
               #data_M2_debug = data_M_debug[1]
               params_m_2 = res2[0]
               m_2 = VIEW_DEPENDENT(data_ALL_debug,None, params_m_2)
           
               #data_M3_debug = data_M_debug[2]
               params_m_3 = res3[0]
               m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_ALL_debug,None, params_m_3)
               
               #ädata_M4_debug = data_M_debug[3]
               params_m_4 = res4[0]
               m_4 = VIEW_INDEPENDENT(data_ALL_debug,None, params_m_4)
           
               #data_M5_debug = data_M_debug[4]
               params_m_5 = res5[0]
               m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_ALL_debug,None, params_m_5)
               
               #data_M6_debug = data_M_debug[5]
               params_m_6 = res6[0]
               m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_ALL_debug,None, params_m_6)
       
               res_debug = {models_names[0]: m_1,
                            models_names[1]: m_2,
                            models_names[2]: m_3,
                            models_names[3]: m_4,
                            models_names[4]: m_5,
                            models_names[5]: m_6}
               data_verbose_debug[i] = res_debug
               
       #### Get winning model trialwise dat ####
           params_m_1 = res1[0]
           res_M1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None, params_m_1)        
   
           trialwise_data[i] = res_M1[1]['data_store_1']
       
       ### some calculation
       restotal = res_evidence.sum(axis=1)
       group_level_me = res_evidence[[i for i in res_evidence]].sum(axis=1)

       
       ##### BIC calc
       indxs = ['bic','raw_LL','n_params', 'sample_size']
       bic_fit= pd.DataFrame(index=indxs)
       for model in models_names_bic:
           n_params = len(list(parameter_est[model].index))
           sample_size = len([i for i in res_evidence])
           raw_LL = group_level_me[model]
           bic_curr = bic(n_params, sample_size, raw_LL)
           bic_fit[model] = [bic_curr,raw_LL,n_params, sample_size]
       
       ## summarize data
       results_1 = {'subject_level_model_evidence':res_evidence,
                   'group_level_me': group_level_me,
                   'group_level_BIC':bic_fit,
                   'used_data': data,
                   'subject_level_parameter-estimates':parameter_est,
                   'subject_level_trialwise_data_win_model':trialwise_data}
       #self.fit_separate = results_1
       if verbose_tot==True:
           return (results_1,restotal,data_verbose_debug)
       elif verbose_tot==False:
           return results_1


def fit_data_base(data_vpn):
    ### data has to be list
    
    from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    
    epsilon_param = .01
    x_0_bfgs = 0.5
    #func calls & rand starts
    vpn = data_vpn[0]
    curr_data_vpn = data_vpn
    # # get data numpy
    stim_IDs_VI=    curr_data_vpn[0:,2]  #stimulus IDs of winning model 
    new_ID=         curr_data_vpn[0:,1]      #trials where new ID is introduced 
    n_prev_pres=    curr_data_vpn[0:,4]    #number_of_prev_presentations
    stim_IDs_VD=    curr_data_vpn[0:,3]  #view dependent
    VPN_output =    curr_data_vpn[0:,0]     #VPN answers
    verbose =       False
    #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

    data_ALL = [VPN_output.astype(int), 
                new_ID.astype(int), 
                n_prev_pres.astype(int), 
                stim_IDs_VI.astype(str), 
                stim_IDs_VD.astype(str), 
                verbose]     
    ##### Data

    params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
    params_M2_name = ['alpha', 'beta', 'lamd_a']
    params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
    params_M4_name = ['alpha', 'beta', 'lamd_a']
    params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
    params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

    models_names = ['VIEW_INDIPENDENTxCONTEXT',
                    'VIEW_DEPENDENT',
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                    'VIEW_INDEPENDENT',
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
                    'random_choice']
    models_names_bic = ['VIEW_INDIPENDENTxCONTEXT',
                        'VIEW_DEPENDENT',
                        'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                        'VIEW_INDEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
    
    parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                    'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                    'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)}   

    
    
    ########## Model Optim
    
    
    print('VIEW_INDIPENDENTxCONTEXT')

    bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
    res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                  approx_grad = True,
                                  bounds = bounds_M1, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDIPENDENTxCONTEXT'][vpn] = res1[0]
    
    ##### VIEW_DEPENDENT
    print('VIEW_DEPENDENT')
            
    bounds_M2 = [(0,1),(.1,20),(0,2)]
    
    part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
    res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                  approx_grad = True,
                                  bounds = bounds_M2, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                  epsilon=epsilon_param)
    
    #parameter_est['VIEW_DEPENDENT'][vpn] = res2[0]
    
    ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
    print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

    bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]

    part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
    res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                  approx_grad = True,
                                  bounds = bounds_M3, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][vpn] = res3[0]
    
    ##### VIEW_INDEPENDENT
    print('VIEW_INDEPENDENT')

    bounds_M4 = [(0,1),(.1,20),(0,2)]

    part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
    res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                  approx_grad = True,
                                  bounds = bounds_M4, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDEPENDENT'][vpn] = res4[0]
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
    print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
            
    bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
    res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                  approx_grad = True,
                                  bounds = bounds_M5, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][vpn] = res5[0]
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
    #params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

    bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL,None) 
    res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                  approx_grad = True,
                                  bounds = bounds_M6, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                  epsilon=epsilon_param)
    
    #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][vpn] = res6[0]
    
    
    ##### RND Choice model
    rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
    
    re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
    
    res_ev_DF = pd.DataFrame(index = models_names)
    res_ev_DF['ev'] = re_evidence_subj
    ### Subject BF_LOG
    bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj))
    
    return res_ev_DF

def generate_data_pwr(path_ground_truth,min_sample,max_sample):
    at_sample_size = 15
    SAMPLE_fullinfo = pd.read_csv(path_ground_truth).drop(columns = ['Unnamed: 0'])
    prev_pres = np.array(SAMPLE_fullinfo[[i for i in SAMPLE_fullinfo][-1]])-1
    fig2_c_mean = [.2,
                   .43,
                   .58,
                   .7,
                   .74,
                   .76,
                   .83,
                   .77,
                   .78,
                   .81,
                   .82,
                   .78]
    fig2_c_SEM = [.09,
                  .08,
                  .07,
                  .09,
                  .09,
                  .08,
                  .09,
                  .1,
                  .05,
                  .07,
                  .07,
                  .09]
    fig2_c_SD = np.array(fig2_c_SEM)*np.sqrt(15)

    res_beh = {}
    res_prob = {}
    for sample_size in range(min_sample,max_sample):
        data_beh_synth = {}
        data_prob_synth = []
        for i in range(sample_size):
            probs = []
            perspective = []
            for trial in prev_pres:
                # generate probability
                curr_prob_raw = np.random.normal(loc=fig2_c_mean[trial], scale=fig2_c_SD[trial])
                # check bounds
                curr_prob = bounds(curr_prob_raw)
                probs.append(curr_prob)
                #generate perspective data
                persp = random.sample(['L','R','M'],1)[0]
                perspective.append(persp)
                
            dat = np.random.binomial(1,probs)
            trials_synth = pd.DataFrame()
            trials_synth['answer'] =        dat
            trials_synth['new_IDs'] =       SAMPLE_fullinfo['new_IDs']
            trials_synth['stim_IDs_VI'] =   SAMPLE_fullinfo['stim_IDs']
            trials_synth['perspective'] =   perspective
            trials_synth['vdstim_raw'] = VD_stim = [str(i)+j for i,j in zip(trials_synth['stim_IDs_VI'],perspective)]
            ##### VD count stim and prev_pres
            unq_stim_VD = np.unique(VD_stim)
            unq_stim_VI = np.unique(trials_synth['stim_IDs_VI'])
            unq_stim_VD_dict = {}
            #new numbering for VD stims
            for key,val in zip(unq_stim_VD,range(len(unq_stim_VD))):
                unq_stim_VD_dict[key] = val
                
            new_key_VD = [unq_stim_VD_dict[i] for i in VD_stim]
            trials_synth['stim_IDs_VD'] = new_key_VD
            ##### prev presentations
            #VD
            prev_pred_dict_VD = dict.fromkeys(np.unique(new_key_VD),0)
            prev_pres_L_VD = []
            #VI
            prev_pred_dict_VI = dict.fromkeys(unq_stim_VI,0)
            prev_pres_L_VI = []
            
            for vi,vd,trial in zip(list(trials_synth['stim_IDs_VI']),list(trials_synth['stim_IDs_VD']),range(len(list(trials_synth['stim_IDs_VD'])))):
                prev_pred_dict_VI[vi] =  prev_pred_dict_VI[vi]+1
                prev_pred_dict_VD[vd] =  prev_pred_dict_VD[vd] +1
                prev_pres_L_VI.append(prev_pred_dict_VI[vi])
                prev_pres_L_VD.append(prev_pred_dict_VD[vd])
            
            trials_synth['n_prev_VI'] =       prev_pres_L_VI
            trials_synth['n_prev_VD'] =       prev_pres_L_VD
            ###clean DF
            trials_synth.drop(axis=1,labels = ['perspective','vdstim_raw'],inplace=True)
            
            ##save data
            data_beh_synth[str(i)] = trials_synth
            data_prob_synth.append(probs)
            
            
            
            
        res_beh[sample_size] = data_beh_synth
        res_prob[sample_size] = data_prob_synth
    return res_beh


def ttest_procedure(subject_level_model_evidence):
    import pandas as pd
    import pingouin as pg
    import numpy as np
    ''' 
    As described in Apps & Tsakiris, we ran a series of independent
    two-sided t-tests and corrected via Benjamini–Hochberg false discovery
    rate (FDR)
    '''
    
    data_fit = subject_level_model_evidence.T
    
    win_model = 'VIEW_INDIPENDENTxCONTEXT'
    indx_models = [i for i in data_fit if 'VIEW_INDIPENDENTxCONTEXT' not in i]
    indx_res = ['p-val','cohen-d','BF10','obs_power','T']
    
    res = pd.DataFrame(index = indx_res)

    #for each time-point

    for model in indx_models:
        res_tt_raw = pg.ttest(data_fit[win_model],data_fit[model])
        res_tt_ind = [res_tt_raw['p-val'][0],
                      res_tt_raw['cohen-d'][0],
                      res_tt_raw['BF10'][0],
                      res_tt_raw['power'][0],
                      res_tt_raw['T'][0]]
        
        clm_name = win_model + '_vs_' + model
        res[clm_name] = res_tt_ind
    #FDR correction
    fdr_dat = res.copy().T.astype(float)
    fdr_dat['pval_FDR'] = pg.multicomp([i for i in fdr_dat['p-val']],
                                         alpha=0.02,
                                         method='fdr_bh')[1]
    return fdr_dat



########### Set WD to where class import file is located

'''
Set current WD where the modules are stored
Note, that the raw-data needs to be placed elsewhere!
'''
import os
path_to_class = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'

os.chdir(path_to_class)


########## Get class import
''' 
This class stores all relevant modules of the analyses mentioned in Zaragoza-Jimenez et al. 2021.
It needs to be in the same folder as all other files
'''
from class_import import Apps_Tsakiris_2013


########## Get list of installes packages and dump as list
import numpy as np

##### location of data
path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'

##### location of ground truth-data
path_ground_truth = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
pathtomodels = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'
data_analyses = Apps_Tsakiris_2013(path_data,
                                   path_ground_truth,
                                   path_to_class)
########## preprocess & clean data
data_processed = data_analyses.get_data(verbose = True)

# Raw LL
data_gen_model = data_analyses.gen_data(False)

path_gen_data = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\synth_data\_model_pred'
for key in data_gen_model.keys():
    curr_dat = data_gen_model[key]
    for counter, value in enumerate(curr_dat):
        print(value)
                                    
    





# path_ground_truth = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
# res123 = generate_data_pwr(path_ground_truth,20,70)

# for i in res123.keys():
#     curr_dat = []
#     for j in res123[i].keys():
#         curr_dat.append(res123[i][j].to_numpy())
#     res123[i] = curr_dat

# res_total = {}
# for i in list(res123.keys()):
#     curr_L = res123[i]
# #if __name__ == 'class_import':   
#     results123 = Parallel(n_jobs=8,
#                           verbose=10,
#                           backend='loky')(delayed(fit_data_base)(k) for k in curr_L)  
#     res_med = pd.DataFrame(index = list(results123[0].index))
#     for counter, value in enumerate(results123):
#         res_med[counter] = value
#     res_tt = ttest_procedure(res_med)
#     win_mod_ev = res_med.loc['VIEW_INDIPENDENTxCONTEXT'].sum()
#     cntrl_mod = [res_med.loc[i].sum() for i in res_med.index if 'random' not in i]
#     post_model_prob = np.exp(win_mod_ev)/np.exp(special.logsumexp(cntrl_mod))
#     res_total[i] = {'subj_ev': res_med,
#                     'mean_obs_power': res_tt['obs_power'].mean(),
#                     'post_model_prob' :post_model_prob}


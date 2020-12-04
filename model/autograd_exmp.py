# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:52:56 2020

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
from class_import import task_icc,pears_corr
import matlab.engine
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad

##### Reformat data for within-time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  


data, lbfgs_epsilon, verbose_tot = data_dict_t1, 0.01, False

def VIEW_INDIPENDENTxCONTEXT(data,params):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    VPN_output = data[0].copy()
    new_ID = data[1]
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    verbose = data[4]
    
    # get false positive answer rate
    #FP_rate = FP_rate_func(numb_prev_presentations, VPN_output, 189)
    trials_FP = len(VPN_output)
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = (FP_rate_raw / trials_FP)*lamd_a

    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_C_pe = []
    history_V_ind = []
    history_total = []
    history_answer = []
    model_evidence = 0
    vfam_PE_list = []

    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        old_c = history_C[-1]
        
        # Update VFam
        vfam_PE = (lamd_a - old_Vfam)
        new_Vfam = old_Vfam + (alpha*vfam_PE)
        vfam_PE_list.append(vfam_PE)
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
        history_V_ind.append(new_Vfam)
        # Update C Fam   
        context_PE = (old_c - old_Vfam)
        new_c = old_c - (sigma * context_PE)
        history_C_pe.append(context_PE)
        history_C.append(new_c)
        
        ### get totfam
        totfam = new_c*new_Vfam    
        history_total.append(totfam)       

        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)+ 0.000001
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
            #model_evidence += np.log(p_yes)
        if action == 0:
            history_answer.append(p_no)
            #model_evidence += np.log(p_no)
    model_evidence = np.log(history_answer.copy()).sum()
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        data_store_1 = pd.DataFrame()
        data_store_1['history_V'] = history_V_ind
        data_store_1['history_C'] = history_C[1::]
        data_store_1['history_total'] = history_total
        data_store_1['vfam_PE'] = vfam_PE_list
        data_store_1['context_PE'] = history_C_pe
        data_store_1['vpn_answer'] = VPN_output

        
        data_store = {'history_answer': history_answer,
                      'params':[alpha, sigma, beta, lamd_a],
                      'log_like': model_evidence,
                      'data_store_1':data_store_1,
                      'history_total':history_total,
                      'init_val':{'init_v': FP_rate,'init_c' : 0}}
        return (model_evidence,data_store)

def VIEW_INDIPENDENTxCONTEXT_grad(data,params):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    VPN_output = data[0].copy()
    new_ID = data[1]
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    verbose = data[4]
    
    # get false positive answer rate
    #FP_rate = FP_rate_func(numb_prev_presentations, VPN_output, 189)
    trials_FP = len(VPN_output)
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = (FP_rate_raw / trials_FP)*lamd_a

    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_C_pe = []
    history_V_ind = []
    history_total = []
    history_answer = []
    model_evidence = 0
    vfam_PE_list = []

    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        old_c = history_C[-1]
        
        # Update VFam
        vfam_PE = (lamd_a - old_Vfam)
        new_Vfam = old_Vfam + (alpha*vfam_PE)
        vfam_PE_list.append(vfam_PE)
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
        history_V_ind.append(new_Vfam)
        # Update C Fam   
        context_PE = (old_c - old_Vfam)
        new_c = old_c - (sigma * context_PE)
        history_C_pe.append(context_PE)
        history_C.append(new_c)
        
        ### get totfam
        totfam = new_c*new_Vfam    
        history_total.append(totfam)       

        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)+ 0.000001
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
            #model_evidence += np.log(p_yes)
        if action == 0:
            history_answer.append(p_no)
            #model_evidence += np.log(p_no)
    model_evidence = np.log(history_answer.copy()).sum()
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        data_store_1 = pd.DataFrame()
        data_store_1['history_V'] = history_V_ind
        data_store_1['history_C'] = history_C[1::]
        data_store_1['history_total'] = history_total
        data_store_1['vfam_PE'] = vfam_PE_list
        data_store_1['context_PE'] = history_C_pe
        data_store_1['vpn_answer'] = VPN_output

        
        data_store = {'history_answer': history_answer,
                      'params':[alpha, sigma, beta, lamd_a],
                      'log_like': model_evidence,
                      'data_store_1':data_store_1,
                      'history_total':history_total,
                      'init_val':{'init_v': FP_rate,'init_c' : 0}}
        return model_evidence

#folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'

#see http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/


import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize,special
np.random.seed(1993)
### Get data 
data = data

#### get unique IDS
unique_id = list(data.keys())
sample_answer_clms = [i+'_answer' for i in unique_id]
sample_perspective_clms = [i+'_perspective' for i in unique_id]

epsilon_param = lbfgs_epsilon
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

for vpn in unique_id:
    print(vpn)
    #func calls & rand starts
    
    curr_data_vpn = data[vpn]
    # data import
    stim_IDs = curr_data_vpn['stim_IDs'] #stimulus IDs of winning model 
    new_ID = curr_data_vpn['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = curr_data_vpn['n_prev_pres'] #number_of_prev_presentations
    stim_IDs_perspective = curr_data_vpn[vpn+'_perspective'] #view dependent
    VPN_output = curr_data_vpn[vpn+'_answer'] #VPN answers
    verbose = False
    
    
    ##### Model Optim
    
    i=vpn
    print('VIEW_INDIPENDENTxCONTEXT')
    data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

    bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
    gradient = grad(part_func_M1)
    gradient1 = gradient((.2,.2,15.,12.5))
    res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                  approx_grad = True,
                                  bounds = bounds_M1, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                  epsilon=epsilon_param)
    parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]  
    
    
    re_evidence_subj = -1*np.array(res1[1])
    res_evidence[i] = re_evidence_subj
    
    ### Subject BF_LOG
    # bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
    # bf_log_group[i + '_BF_log'] = [bf_log_subj]
    
    
    # trialwise_dat = {}
    ############################## Verbose == True ###########################
    
    verbose_debug = verbose_tot
    
    data_M_debug = [data_M1]
    for dat in data_M_debug:
        dat[-1] = True        
    
    if verbose_debug == True:

        data_M1_debug = data_M_debug[0]
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
 
    

        res_debug = {models_names[0]: m_1}
        data_verbose_debug[i] = res_debug
        
#### Get winning model trialwise dat ####
    data_M1_debug = data_M_debug[0]
    params_m_1 = res1[0]
    m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)        

    trialwise_data[i] = m_1[1]['data_store_1']
restotal = res_evidence.sum(axis=1)
cntrl_log = special.logsumexp(np.array(restotal[1::]))
bf_log = (cntrl_log -(np.array(restotal[0])))

results_1 = {'uncorr_LR_10':np.exp(-1*bf_log),
            'subject_level_model_evidence':res_evidence,
            'group_level_model_evidence':res_evidence.sum(axis=1),
            'subject_level_uncorr_LR': bf_log_group,
            'xxx':data_verbose_debug,
            'used_data': data,
            'subject_level_parameter-estimates':parameter_est,
            'subject_level_trialwise_data_win_model':trialwise_data}
# if verbose_tot==True:
#     return (results_1,restotal,data_verbose_debug)
# elif verbose_tot==False:
#     return results_1
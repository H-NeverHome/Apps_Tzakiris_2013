# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 08:41:08 2021

@author: de_hauk
"""

import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT
from class_import import fit_data_noCV_irr_len_data,data_fit_t1_t2_comb
from class_import import bayes_RFX_cond,orig_procedure
from class_import import reformat_data_within_T,fit_data_CV,bic_modelselect
from class_import import task_rel,corr_lr_func,fit_data_CV_mult,get_data_3
import matlab.engine
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 
from joblib import Parallel, delayed

    
def VIEW_INDIPENDENTxCONTEXT(data, cv_trial, params):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    VPN_output = data[0].copy()
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    verbose = data[5]
    
    # get false positive answer rate
    FP_rate = FP_rate_independent(lamd_a,VPN_output,numb_prev_presentations)

    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_C_pe = []
    history_V_ind = []
    history_total = []
    history_answer = []
    model_evidence = 0
    vfam_PE_list = []

    action_CV_L = []
    ### inner loop through trials
    #for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
    for trial in range(len(stim_IDs)):
       
        stim_ID,action = stim_IDs[trial],VPN_output[trial]
        
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        old_c = history_C[-1]   
        
        
        # Update VFam
        new_Vfam,vfam_PE = update_view_independent(lamd_a,alpha,old_Vfam)
        vfam_PE_list.append(vfam_PE)
        
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
        history_V_ind.append(new_Vfam)
        # Update C Fam   
        new_c, context_PE = update_context(sigma,old_c,old_Vfam)
        history_C_pe.append(context_PE)
        history_C.append(new_c)
        
        ### get totfam
        totfam = new_c*new_Vfam    
        history_total.append(totfam)       

        #get answer prob
        p_yes,p_no = answer_prob(beta,totfam)
        if cv_trial == None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
            action_CV_L.append(action)
        elif cv_trial != None:
            if cv_trial != trial:
                if action == 1:
                    history_answer.append(p_yes)
                if action == 0:
                    history_answer.append(p_no)
                action_CV_L.append(action)
            else:
                action_CV = np.random.binomial(1,np.around(p_yes,decimals=5),1)[0]
                action_CV_L.append(action_CV)
            
    model_evidence = np.log(history_answer).sum()
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        
        model_internals = {'answer_prob':       history_answer,
                           'init_val':          {'init_v': FP_rate,'init_c' : 0},
                           'history_V_dict':    history_V,
                           'history_totfam':    history_total,
                           'history_V' :        history_V_ind,
                           'history_C' :        history_C[1::],
                           'vfam_PE' :          vfam_PE_list,
                           'context_PE' :       history_C_pe,
                           'action original' :  VPN_output,
                           'action_CV':         action_CV_L
                           }
        
        model_res={'params':[alpha, sigma, beta, lamd_a],
                   'log_like': model_evidence,
                   'model_internals':model_internals}

        return (model_evidence,model_res)
    
################################### Calc False-Positive Rate as initialization val #################################
########## View-independent FP Rate

def FP_rate_independent(lamd_a,VPN_output,numb_presentations):
    trials_FP = len(VPN_output)
    FP_rate_raw = 0
    for i,j in zip(numb_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = (FP_rate_raw / trials_FP)*lamd_a
    return FP_rate

########## View-dependent FP Rate
def FP_rate_dependent(lamd_a,VPN_output,numb_presentations):
    # get view-dependent false positive answer rate
    trials_FP = len(VPN_output)
    FP_rate_raw = 0
    for n_pres,answ in zip(numb_presentations, VPN_output):
        if n_pres==1 and answ==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = (FP_rate_raw / trials_FP)*lamd_a
    return FP_rate

################################### Updating equations & Helpers
########## View-independence

def update_view_independent(lambd_a,alpha,old_vfam):
    vfam_PE = (lambd_a - old_vfam)
    new_Vfam = old_vfam + (alpha*vfam_PE)
    return float(new_Vfam),float(vfam_PE)

########## Context
def update_context(sigma,old_c,old_vfam):
    context_PE = (old_c - old_vfam)
    new_c = old_c - (sigma * context_PE)
    return float(new_c),float(context_PE)

########## View-dependence
def update_view_dependent(lambd_a,alpha,old_vfam):
    vfam_PE = lambd_a-old_vfam
    new_fam = old_vfam + (alpha * vfam_PE) # compute new stim familiarity
    return float(new_fam),float(vfam_PE)

########## Answer prob
def answer_prob(beta,totfam):
    p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)+ 0.000001
    p_no = np.around((1-p_yes), decimals=5) + 0.000001
    return p_yes,p_no   


########## View_dependent suppl. Data

def view_dep_suppl_dat(stim_IDs_perspective):
    
    # Get unique amount of stimuli respecting presentation angle
    all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
    unq_stim = np.unique(all_stim,return_counts = True)

    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in all_stim:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    return unq_stim,numb_presentations

    
    

#folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'



data_2_sample = get_data_2(r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

data_3_sample = get_data_3(r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

##### Reformat data for within-time format
data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  



import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize,special
np.random.seed(1993)
### Get data 
data = data_3_sample['A']

#### get unique IDS
unique_id = list(data.keys())
sample_answer_clms = [i+'_answer' for i in unique_id]
sample_perspective_clms = [i+'_perspective' for i in unique_id]

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

parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                    }
models_bounds = {'VIEW_INDIPENDENTxCONTEXT':[(0.,1.),(0.,1.),(.1,20.),(0.,2.)],
                'VIEW_DEPENDENT': [],
                'VIEW_DEPENDENTxCONTEXT_DEPENDENT':[],
                'VIEW_INDEPENDENT':[],
                'VIEW_INDEPENDENTxVIEW_DEPENDENT':[],
                'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT':[],
                'random_choice': []}
                
data_verbose_debug = {}
res_evidence = pd.DataFrame(index=models_names)
trialwise_data = {}
bf_log_group = pd.DataFrame()
res_all = []
for vpn in unique_id[0:1]:
    print(vpn)

    curr_data_vpn = data[vpn]
    
    # get data
    stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
    new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
    n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
    stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
    VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
    verbose =       False
    
    data_ALL = [VPN_output.astype(int), 
                new_ID.astype(int), 
                n_prev_pres.astype(int), 
                stim_IDs_VI, 
                stim_IDs_VD, 
                verbose]
    data_ALL_verbose = data_ALL.copy()
    data_ALL_verbose[-1] = True
    
    
    final_data_data = []
    
    bounds_M1 = models_bounds['VIEW_INDIPENDENTxCONTEXT']
        
    part_func_M1A = partial(VIEW_INDIPENDENTxCONTEXT, data_ALL, None)
    # optimize model
    res11 = optimize.fmin_l_bfgs_b(part_func_M1A,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        
    expl_res11 = VIEW_INDIPENDENTxCONTEXT(data_ALL_verbose, None, res11[0])

    for trials_CV in range(len(VPN_output)):
        print(trials_CV)
        ##### Model Optim
        #print('VIEW_INDIPENDENTxCONTEXT')
    
    
        bounds_M1 = models_bounds['VIEW_INDIPENDENTxCONTEXT']
        
        part_func_M1A = partial(VIEW_INDIPENDENTxCONTEXT, data_ALL, None)
        # optimize model
        res11 = optimize.fmin_l_bfgs_b(part_func_M1A,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        
        expl_res11 = VIEW_INDIPENDENTxCONTEXT(data_ALL_verbose, None, res11[0])
        
        
        # optimize model CV
        
        trials_CV = 55
        part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT, data_ALL, trials_CV)
        
        res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        
        expl_res = VIEW_INDIPENDENTxCONTEXT(data_ALL_verbose, trials_CV, res1[0])
        #evaluate model
        data_ALL_CV = [expl_res[1]['model_internals']['action_CV'],#VPN_output.astype(int), 
                       new_ID.astype(int), 
                       n_prev_pres.astype(int), 
                       stim_IDs_VI, 
                       stim_IDs_VD, 
                       True]
        
        expl_res_CV = VIEW_INDIPENDENTxCONTEXT(data_ALL_CV, None, res1[0])
        cv_score_log = np.log(expl_res_CV[1]['model_internals']['answer_prob'][trials_CV])
        
        res_total_fin = {'opt_norm': expl_res11,
                         'gen_CV_dat': expl_res,
                         'CV_model': expl_res_CV,
                         'CV_score': cv_score_log}
        final_data_data.append(cv_score_log)
    #TODO Wrong results
    final_data_sum = np.sum(final_data_data)   
    #res_all.append((res1,expl_res,expl_res1))

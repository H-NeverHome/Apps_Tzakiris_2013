# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:22:16 2020

@author: hauke
"""

import numpy as np
################################### Calc False-Positive Rate as initialization val #################################

def FP_rate_func(N_presentations, curr_list, trials):
    FP_rate_raw = 0
    for i,j in zip(N_presentations, curr_list):
        if i==1 and j==1:
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials
    return FP_rate

################################### Winning Model: INDIPENDENTxCONTEXT #############################################
# partial // Argumentabfolge functools partials 
# donald knut


def VIEW_INDIPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, verbose):
    
    # get false positive answer rate
    #FP_rate = FP_rate_func(numb_prev_presentations, VPN_output, 189)
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP

    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_V_ind = []
    history_total = []
    history_answer = []
    model_evidence = 0
    
    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        old_c = history_C[-1]
        
        # Update VFam
        err_V = lamd_a - old_Vfam
        w_err_V = alpha*err_V
        new_Vfam = old_Vfam + w_err_V
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
        
        # Update C Fam
        err_C = old_c - old_Vfam
        w_err_C = sigma * err_C    
        new_c = old_c - w_err_C
        history_C.append(new_c)
        
        ### get totfam
        totfam = new_c*new_Vfam    
        history_total.append(totfam)       
        
        ### debugging
        
        history_V_ind.append((old_Vfam,new_Vfam))
        history_total.append(totfam)
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
            model_evidence += np.log(p_yes)
        if action == 0:
            model_evidence += np.log(p_no)
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    if verbose == False:
        return model_evidence
    elif verbose == True:
        data_store = {'history_answer': history_answer,
              'history_V': history_V,
              'history_total':history_total,
              'history_C': history_C,
              'params':[alpha, sigma, beta, lamd_a],
              'log_like': model_evidence}
        return (model_evidence,data_store)
   

################################### Control Model: VIEW_DEPENDENT #############################################

def VIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose):
    ''' Each stimulus presented from each of the 3 perspectives constitutes a new stimulus i.e. 24*3 Perspectives <= 72 possible stim IDs. BUT perspective is random - not necessary 72 unq stim 
    Stimulus context does not play a role.'''
    
    # Get unique amount of stimuli respecting presentation angle
    all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
    unq_stim = np.unique(all_stim,return_counts = True)
    
    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in stim_IDs_perspective:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    
    # get false positive answer rate
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP

   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R

    history_answer = []
    history_total = []
   
    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
       
        ### Get all data for current trial
        old_fam = history_V_depend[stim_ID][-1] # previos stim familiarity
        
        ### Update view_dep familarity
        new_fam = old_fam + (alpha * (lamd_a-old_fam)) # compute new stim familiarity
        
        # protocol data
        newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
        history_V_depend[stim_ID] = newfamlist # replace list in dict
           
        ### get totfam @ time t
        totfam = new_fam    
        history_total.append(totfam)
    
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    if verbose == True:
        data_store = {'history_answer': history_answer,
                      'history_V': history_V_depend,
                      'history_total':history_total,
                      'params':[alpha, beta, lamd_a],
                      'log_like': model_evidence}
        return (model_evidence,data_store)
    elif verbose == False:
         return (model_evidence)


################################### Control Model: VIEW_DEPENDENTxCONTEXT_DEPENDENT MODEL #############################################

def VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose):

    # Get unique amount of stimuli
    all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
    unq_stim = np.unique(all_stim,return_counts = True)
    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in stim_IDs_perspective:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    


    # get false positive answer rate
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_V_dep = []
    history_total = []
    history_answer = []
    model_evidence = 0
    
   
    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
       
        ### Get all data for current trial
        old_Vfam = history_V_depend[stim_ID][-1] # previos stim familiarity
        old_c = history_C[-1]
        
        # Update C Fam
        err_C = old_c - old_Vfam
        w_err_C = sigma * err_C    
        new_c = old_c - w_err_C

        
        ### Update view_dep familarity
        new_fam = old_Vfam + (alpha * (lamd_a-old_Vfam)) # compute new stim familiarity
        
        # protocol data
        newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
        history_V_depend[stim_ID] = newfamlist # replace list in dict
        history_C.append(new_c) 

        ### get totfam @ time t
        totfam = new_fam * new_c 
        
        history_V_dep.append((old_Vfam,new_fam))
        history_total.append(totfam)

        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001 # implemented constant as to avoid np.log(0)
        #print(p_yes,p_no)
        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V': history_V_depend,
                  'history_total':history_total,
                  'history_C': history_C,
                  'params':[alpha, sigma, beta, lamd_a],
                  'log_like': model_evidence}
    if verbose == False:
        return model_evidence
    elif verbose == True:
        return (model_evidence,data_store)

# ################################### Control Model: VIEW_INDEPENDENT MODEL #############################################


def VIEW_INDEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose):
    
    # get false positive answer rate
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP
    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_total = []
    history_answer = []
    model_evidence = 0
    
    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        
        # Update VFam
        err_V = lamd_a - old_Vfam
        w_err_V = alpha*err_V
        new_Vfam = old_Vfam + w_err_V

        ### get totfam
        totfam = new_Vfam    
        history_total.append(totfam)
        ### Update C & V
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
    
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V': history_V,
                  'history_total':history_total,
                  'params':[alpha, beta, lamd_a],
                  'log_like': model_evidence}
    if verbose == False:
        return model_evidence
    elif verbose == True:
        return (model_evidence,data_store)     

# @utils.use_named_args(dimensions=dimensions_wo_context)
# def VIEW_INDEPENDENT_optim(alpha, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs)
#     model_ev = result[0]
#     return -1*model_ev                                                              

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODEL #############################################  
def VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose):
    
    # Get unique amount of stimuli
    all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
    unq_stim = np.unique(all_stim,return_counts = True)
    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in stim_IDs_perspective:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    
    # get false positive answer rate
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate])
    history_answer = []
    history_total = []
    
   
    ### inner loop through trials
    for stim_ID_depend, stim_ID_indipend, action,num_pres_depend, num_pres_independ,trial in zip(stim_IDs_perspective,stim_IDs,VPN_output,numb_presentations,numb_prev_presentations,range(len(stim_IDs))):
       
        ### Get all data for current trial
        old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
        old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
        
        # update view-independent
        new_fam_ind = old_fam_indipend + (alpha_ind * (lamd_a_ind - old_fam_indipend))
        # update view-dependent
        new_fam_dep = old_fam_depend + (alpha_dep * (lamd_a_dep - old_fam_depend)) # compute new stim familiarity
                
        #protocol data
        new_fam_indipend_list = history_V_independ[stim_ID_indipend].copy() + [new_fam_ind] # append new stim familiarity to list
        new_fam_depend_list = history_V_depend[stim_ID_depend].copy() + [new_fam_dep]
        history_V_depend[stim_ID_depend] = new_fam_depend_list # replace list in dict
        history_V_independ[stim_ID_indipend] = new_fam_indipend_list


        ### get totfam @ time t
        totfam = new_fam_ind + new_fam_dep   
        history_total.append(totfam)
    
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_depend': history_V_depend,
                  'history_V_independ': history_V_independ,
                  'history_total': history_total,
                  'params': [alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep],
                  'log_like': model_evidence}
    if verbose == False:
        return model_evidence
    elif verbose == True:
        return (model_evidence, data_store)  


# @utils.use_named_args(dimensions=dimensions_wo_context)
# def VIEW_INDEPENDENTxVIEW_DEPENDENT_optim(alpha, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODELxCONTEXT MODEL #############################################
    
def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose):
    
    # Get unique amount of stimuli
    all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
    unq_stim = np.unique(all_stim,return_counts = True)
    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in stim_IDs_perspective:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    
    # get false positive answer rate
    trials_FP = 189
    FP_rate_raw = 0
    for i,j in zip(numb_prev_presentations, VPN_output):
        if i==1 and j==1: # check if incorrectly assumed known at first presentation
            FP_rate_raw +=1
    FP_rate = FP_rate_raw / trials_FP
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate])
    history_C = [0]

    history_total = []
    history_answer = []
    model_evidence = 0
    
   
    ### inner loop through trials
    for stim_ID_depend, stim_ID_indipend, action,num_pres_depend, num_pres_independ,trial in zip(stim_IDs_perspective,stim_IDs,VPN_output,numb_presentations,numb_prev_presentations,range(len(stim_IDs))):
       
        ### Get all data for current trial
        
        old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
        old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
        old_c = history_C[-1]
        
        # update view-independent
        new_fam_ind = old_fam_indipend + (alpha_ind * (lamd_a_ind - old_fam_indipend))
        # update view-dependent
        new_fam_dep = old_fam_depend + (alpha_dep * (lamd_a_dep - old_fam_depend)) # compute new stim familiarity
        # update context
        new_c = old_c - (sigma *(old_c - (old_fam_depend + old_fam_indipend)))
                
        new_fam_indipend_list = history_V_independ[stim_ID_indipend].copy() + [new_fam_ind] # append new stim familiarity to list
        new_fam_depend_list = history_V_depend[stim_ID_depend].copy() + [new_fam_dep]
        
        history_V_depend[stim_ID_depend] = new_fam_depend_list # replace list in dict
        history_V_independ[stim_ID_indipend] = new_fam_indipend_list
        history_C.append(new_c)
        
        ### get totfam @ time t
        totfam = (new_fam_ind + new_fam_dep) * new_c
        history_total.append(totfam)
    
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001 # implemented constant as to avoid np.log(0)

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
            
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_depend': history_V_depend,
                  'history_V_independ': history_V_independ,
                  'history_C':history_C,
                  'history_total': history_total,
                  'params': [alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep],
                  'log_like': model_evidence}
    if verbose == False:
        return model_evidence
    elif verbose == True:
        return (model_evidence, data_store)
    
# @utils.use_named_args(dimensions=dimensions)
# def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim(alpha, sigma, beta, lamd_a):
#         ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev


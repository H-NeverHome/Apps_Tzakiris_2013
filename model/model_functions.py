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


def VIEW_INDIPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs):
    
    # get false positive answer rate
    FP_rate = FP_rate_func(numb_prev_presentations, VPN_output, 189)
    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_total = []
    history_answer = []
    model_evidence = 0
    
    ### inner loop through trials
    for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        #print(stim_ID,action,num_pres,trial)
        
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        old_c = history_C[-1]
        totfam_last = old_Vfam*old_c
        
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
    
        #get answer prob
        p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
        p_no = np.around((1-p_yes), decimals=5) + 0.000001

        if action == 1:
            history_answer.append(p_yes)
            model_evidence += np.log(p_yes)
        if action == 0:
            model_evidence += np.log(p_no)
    return (model_evidence, alpha, sigma, beta, lamd_a)     

# @utils.use_named_args(dimensions=dimensions)
# def VIEW_INDIPENDENTxCONTEXT_optim(alpha, sigma, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDIPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs)
#     model_ev = result[0]
#     return -1*model_ev                                                                

# ################################### Control Model: VIEW_DEPENDENT #############################################

# def VIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective):
#     ''' Each stimulus presented from each of the 3 perspectives constitutes a new stimulus i.e. 24*3 Perspectives <= 72 possible stim IDs. BUT perspective is random - not necessary 72 unq stim 
#     Stimulus context does not play a role.'''
    
#     # Get unique amount of stimuli
#     all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
#     unq_stim = np.unique(all_stim,return_counts = True)
#     # count occurences of stimuli
#     new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
#     numb_presentations = [] # append number of presentations per stim over course off experiment
#     for i in stim_IDs_perspective:
#         new_numb_presentations[i] +=1
#         numb_presentations.append(new_numb_presentations[i])
    


#     # get false positive answer rate
#     FP_rate = FP_rate_func(numb_presentations, VPN_output, 189)

   
#     ### dict for Trackkeeping of history
#     history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R

#     history_total = []
#     history_answer = []
#     model_evidence = 0
    
   
#     ### inner loop through trials
#     for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
       
#         ### Get all data for current trial
        
#         old_fam = history_V_depend[stim_ID][-1] # previos stim familiarity
        
#         new_fam = old_fam + (alpha * (lamd_a-old_fam)) # compute new stim familiarity
        
#         newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
#         history_V_depend[stim_ID] = newfamlist # replace list in dict
        
        
#         history_total.append((old_fam,new_fam)) # protocol change for debugging
        
        

#         ### get totfam @ time t
#         totfam = new_fam    
#         history_total.append(totfam)
    
#         #get answer prob
#         p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
#         p_no = np.around((1-p_yes), decimals=5) + 0.000001

#         if action == 1:
#             history_answer.append(p_yes)
#             model_evidence += np.log(p_yes)
#         if action == 0:
#             model_evidence += np.log(p_no)
#     #return(unq_stim,history_V_depend, history_total,numb_presentations)
#     return (model_evidence, alpha, beta, lamd_a) 

# @utils.use_named_args(dimensions=dimensions_wo_context)
# def VIEW_DEPENDENT_optim(alpha, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev

# ################################### Control Model: VIEW_DEPENDENTxCONTEXT_DEPENDENT MODEL #############################################

# def VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective):

    
#     # Get unique amount of stimuli
#     all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
#     unq_stim = np.unique(all_stim,return_counts = True)
#     # count occurences of stimuli
#     new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
#     numb_presentations = [] # append number of presentations per stim over course off experiment
#     for i in stim_IDs_perspective:
#         new_numb_presentations[i] +=1
#         numb_presentations.append(new_numb_presentations[i])
    


#     # get false positive answer rate
#     FP_rate = FP_rate_func(numb_presentations, VPN_output, 189)

   
#     ### dict for Trackkeeping of history
#     history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
#     history_C = [0]
    
#     history_total = []
#     history_answer = []
#     model_evidence = 0
    
   
#     ### inner loop through trials
#     for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
       
#         ### Get all data for current trial
        
#         old_fam = history_V_depend[stim_ID][-1] # previos stim familiarity
#         old_c = history_C[-1]
        
#         new_fam = old_fam + (alpha * (lamd_a-old_fam)) # compute new stim familiarity
#         new_c = model.update_c_fam(old_c,old_fam, sigma) # compute new context familiarity
        
#         newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
#         history_V_depend[stim_ID] = newfamlist # replace list in dict
#         history_C.append(new_c)
        
#         history_total.append((old_fam,new_fam)) # protocol change for debugging
        
        

#         ### get totfam @ time t
#         totfam = new_fam * new_c    
#         history_total.append(totfam)
    
#         #get answer prob
#         p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
#         p_no = np.around((1-p_yes), decimals=5) + 0.000001 # implemented constant as to avoid np.log(0)
#         #print(p_yes,p_no)
#         if action == 1:
#             history_answer.append(p_yes)
#             model_evidence += np.log(p_yes)
#         if action == 0:
#             model_evidence += np.log(p_no)
#     #return(unq_stim,history_V_depend, history_total,numb_presentations)
#     return (model_evidence, alpha,sigma, beta, lamd_a,history_V_depend, history_C) 

# @utils.use_named_args(dimensions=dimensions)
# def VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim(alpha, sigma, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev

# ################################### Control Model: VIEW_INDEPENDENT MODEL #############################################


# def VIEW_INDEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs):
    
#     # get false positive answer rate
#     FP_rate = FP_rate_func(numb_prev_presentations, VPN_output, 189)
    
#     ### dict for Trackkeeping of history
#     history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
#     history_total = []
#     history_answer = []
#     model_evidence = 0
    
#     ### inner loop through trials
#     for stim_ID,action,num_pres,trial in zip(stim_IDs,VPN_output,numb_prev_presentations,range(len(stim_IDs))):
        
#         ### Get V, C & totfam for current trial
#         old_fam = history_V[stim_ID][-1]
#         new_fam = model.update_stim_fam(stim_ID,old_fam,alpha,lamd_a,num_pres)
    
#         ### get totfam
#         totfam = old_fam    
#         history_total.append(totfam)
#         ### Update C & V
#         newfamlist = history_V[stim_ID].copy() + [new_fam]
#         history_V[stim_ID] = newfamlist
    
#         #get answer prob
        
#         p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
#         p_no = np.around((1-p_yes), decimals=5) + 0.000001

#         if action == 1:
#             history_answer.append(p_yes)
#             model_evidence += np.log(p_yes)
#         if action == 0:
#             model_evidence += np.log(p_no)
#     return (model_evidence, alpha, beta, lamd_a)       

# @utils.use_named_args(dimensions=dimensions_wo_context)
# def VIEW_INDEPENDENT_optim(alpha, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs)
#     model_ev = result[0]
#     return -1*model_ev                                                              

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODEL #############################################
    
# def VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective):
    
#     # Get unique amount of stimuli
#     all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
#     unq_stim = np.unique(all_stim,return_counts = True)
#     # count occurences of stimuli
#     new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
#     numb_presentations = [] # append number of presentations per stim over course off experiment
#     for i in stim_IDs_perspective:
#         new_numb_presentations[i] +=1
#         numb_presentations.append(new_numb_presentations[i])
    


#     # get false positive answer rate
#     FP_rate = FP_rate_func(numb_presentations, VPN_output, 189)

   
#     ### dict for Trackkeeping of history
#     history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
#     history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate])
#     history_total = []
#     history_answer = []
#     model_evidence = 0
    
   
#     ### inner loop through trials
#     for stim_ID_depend, stim_ID_indipend, action,num_pres_depend, num_pres_independ,trial in zip(stim_IDs_perspective,stim_IDs,VPN_output,numb_presentations,numb_prev_presentations,range(len(stim_IDs))):
       
#         ### Get all data for current trial
        
#         old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
#         old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
        
#         new_fam_indipend = model.update_stim_fam(stim_ID_indipend,old_fam_indipend,alpha,lamd_a,num_pres_independ)
#         new_fam_depend = old_fam_depend + (alpha * (lamd_a-old_fam_depend)) # compute new stim familiarity
        
#         new_fam_indipend_list = history_V_independ[stim_ID_indipend].copy() + [new_fam_indipend] # append new stim familiarity to list
#         new_fam_depend_list = history_V_depend[stim_ID_depend].copy() + [new_fam_depend]
        
#         history_V_depend[stim_ID_depend] = new_fam_depend_list # replace list in dict
#         history_V_independ[stim_ID_indipend] = new_fam_indipend_list
        
#         #history_total.append((old_fam,new_fam)) # protocol change for debugging
        
        

#         ### get totfam @ time t
#         totfam = old_fam_indipend + old_fam_depend   
#         history_total.append(totfam)
    
#         #get answer prob
#         p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
#         p_no = np.around((1-p_yes), decimals=5) + 0.000001

#         if action == 1:
#             history_answer.append(p_yes)
#             model_evidence += np.log(p_yes)
#         if action == 0:
#             model_evidence += np.log(p_no)
#     #return(unq_stim,history_V_depend, history_total,numb_presentations)
#     return (model_evidence, alpha, beta, lamd_a,history_V_depend, history_V_independ)

# @utils.use_named_args(dimensions=dimensions_wo_context)
# def VIEW_INDEPENDENTxVIEW_DEPENDENT_optim(alpha, beta, lamd_a):
#     ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODELxCONTEXT MODEL #############################################
    
# def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective):
    
#     # Get unique amount of stimuli
#     all_stim = [i[0:4] for i in stim_IDs_perspective if len(i) == 4]
#     unq_stim = np.unique(all_stim,return_counts = True)
#     # count occurences of stimuli
#     new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
#     numb_presentations = [] # append number of presentations per stim over course off experiment
#     for i in stim_IDs_perspective:
#         new_numb_presentations[i] +=1
#         numb_presentations.append(new_numb_presentations[i])
    
#     # get false positive answer rate
#     FP_rate = FP_rate_func(numb_presentations, VPN_output, 189)

#     ### dict for Trackkeeping of history
#     history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
#     history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate])
#     history_C = [0]

#     history_total = []
#     history_answer = []
#     model_evidence = 0
    
   
#     ### inner loop through trials
#     for stim_ID_depend, stim_ID_indipend, action,num_pres_depend, num_pres_independ,trial in zip(stim_IDs_perspective,stim_IDs,VPN_output,numb_presentations,numb_prev_presentations,range(len(stim_IDs))):
       
#         ### Get all data for current trial
        
#         old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
#         old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
#         old_c = history_C[-1]
        
#         new_fam_indipend = model.update_stim_fam(stim_ID_indipend,old_fam_indipend,alpha,lamd_a,num_pres_independ)
#         new_fam_depend = old_fam_depend + (alpha * (lamd_a-old_fam_depend)) # compute new stim familiarity
        
#         err = (old_c - (old_fam_indipend + old_fam_depend) )
#         p_err = sigma * float(err)
#         new_c = old_c - p_err
                
#         new_fam_indipend_list = history_V_independ[stim_ID_indipend].copy() + [new_fam_indipend] # append new stim familiarity to list
#         new_fam_depend_list = history_V_depend[stim_ID_depend].copy() + [new_fam_depend]
        
#         history_V_depend[stim_ID_depend] = new_fam_depend_list # replace list in dict
#         history_V_independ[stim_ID_indipend] = new_fam_indipend_list
#         history_C.append(new_c)
#         #history_total.append((old_fam,new_fam)) # protocol change for debugging
        
#         ### get totfam @ time t
#         totfam = (old_fam_indipend + old_fam_depend) * old_c
#         history_total.append(totfam)
    
#         #get answer prob
#         p_yes = np.around((1/ (1+ np.exp((-1*beta)*totfam))), decimals=5)
#         p_no = np.around((1-p_yes), decimals=5) + 0.000001 # implemented constant as to avoid np.log(0)

#         if action == 1:
#             history_answer.append(p_yes)
#             model_evidence += np.log(p_yes)
#         if action == 0:
#             model_evidence += np.log(p_no)
#     #return(unq_stim,history_V_depend, history_total,numb_presentations)
#     return (model_evidence, alpha,sigma, beta, lamd_a,history_V_depend, history_V_independ,history_C)

# @utils.use_named_args(dimensions=dimensions)
# def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim(alpha, sigma, beta, lamd_a):
#         ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
#     result = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
#     model_ev = result[0]
#     return -1*model_ev


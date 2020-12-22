# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:22:16 2020

@author: hauke
"""

import numpy as np
import pandas as pd
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



################################### Winning Model: INDIPENDENTxCONTEXT #############################################
# partial // Argumentabfolge functools partials 
# donald knut

#params [alpha, sigma, beta, lamd_a,]
#data [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
# verbose

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
    FP_rate = FP_rate_independent(lamd_a,VPN_output,numb_prev_presentations)

    
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

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
            
    model_evidence = np.log(history_answer).sum()
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
   
##### CV

def VIEW_INDIPENDENTxCONTEXT_CV(params,old_vfam,old_cfam,action):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    old_Vfam = old_vfam
    old_c = old_cfam
    action = action

    # Update VFam
    new_Vfam,vfam_PE = update_view_independent(lamd_a,alpha,old_Vfam)

    # Update C Fam   
    new_c, context_PE = update_context(sigma,old_c,old_Vfam)
    
    ### get totfam
    totfam = new_c*new_Vfam 
    
    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)
    if action == 1:
        return np.log(p_yes)
    if action == 0:
        return np.log(p_no)


################################### Control Model: VIEW_DEPENDENT #############################################
#params = [alpha, beta, lamd_a]
#data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]

def VIEW_DEPENDENT(data, params):
    alpha, beta, lamd_a = params[0], params[1], params[2]
    VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[2],data[3],data[4]
    
    ''' Each stimulus presented from each of the 3 perspectives constitutes a new stimulus i.e. 24*3 Perspectives <= 72 possible stim IDs. BUT perspective is random - not necessary 72 unq stim 
    Stimulus context does not play a role.'''
    
    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)

    FP_rate_dep = FP_rate_dependent(lamd_a,VPN_output,numb_presentations)
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R

    history_answer = []
    history_total = []
    history_V_tot = []
    ### inner loop through trials
    #for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
    for trial in range(len(stim_IDs)):
  
        stim_ID,action,num_pres = stim_IDs_perspective[trial], VPN_output[trial], numb_presentations[trial]
        ### Get all data for current trial
        old_fam = history_V_depend[stim_ID][-1] # previos stim familiarity
        
        ### Update view_dep familarity
        new_fam,vfam_PE = update_view_dependent(lamd_a,alpha,old_fam)# compute new stim familiarity
        
        ### get totfam @ time t
        totfam = new_fam    
        
        # protocol data
        newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
        history_V_depend[stim_ID] = newfamlist # replace list in dict
        history_total.append(totfam)
        history_V_tot.append(new_fam)
        
        # get answer prob
        p_yes,p_no = answer_prob(beta,totfam)

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    if verbose == True:
        data_store = {'history_answer': history_answer,
                      'history_V_cv' : history_V_tot,
                      'history_V': history_V_depend,
                      'history_total':history_total,
                      'params':[alpha, beta, lamd_a],
                      'log_like': model_evidence,
                      'suppl': [FP_rate_dep,numb_presentations, VPN_output],
                      'init_val': FP_rate_dep}
        return (model_evidence,data_store)
    elif verbose == False:
         return -1*model_evidence

def VIEW_DEPENDENT_CV(params, old_fam, action):
    # ML Parameters 
    alpha, beta, lamd_a = params[0], params[1], params[2]
    
    ### Update view_dep familarity
    new_fam, vfam_PE = update_view_dependent(lamd_a,alpha,old_fam) # compute new stim familiarity
       
    ### get totfam @ time t
    totfam = new_fam    

    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)
    if action == 1:
        return np.log(p_yes)
    if action == 0:
        return np.log(p_no)
    


################################### Control Model: VIEW_DEPENDENTxCONTEXT_DEPENDENT MODEL #############################################
#params = [alpha, sigma, beta, lamd_a,]
#data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]


def VIEW_DEPENDENTxCONTEXT_DEPENDENT(data, params):
    
    # Data & Parameter
    alpha, sigma, beta, lamd_a = params[0], params[1], params[2], params[3],
    VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[2],data[3],data[4]
    

    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)
   
    FP_rate = FP_rate_dependent(lamd_a,VPN_output,numb_presentations)
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_C = [0]
    history_V_dep = []
    history_total = []
    history_answer = []
    model_evidence = 0
    
   
    ### inner loop through trials
    #for stim_ID,action,num_pres,trial in zip(stim_IDs_perspective,VPN_output,numb_presentations,range(len(stim_IDs))):
    for trial in range(len(stim_IDs)):
        stim_ID,action,num_pres = stim_IDs_perspective[trial],VPN_output[trial],numb_presentations[trial]
        
        ### Get all data for current trial
        old_Vfam = history_V_depend[stim_ID][-1] # previos stim familiarity
        old_c = history_C[-1]

        ### Update view_dep familarity
        new_fam,vfam_PE = update_view_dependent(lamd_a,alpha,old_Vfam) # compute new stim familiarity
        
        # Update C Fam   
        new_c,c_PE = update_context(sigma,old_c,old_Vfam)       
        # protocol data
        newfamlist = history_V_depend[stim_ID].copy() + [new_fam] # append new stim familiarity to list
        history_V_depend[stim_ID] = newfamlist # replace list in dict
        history_C.append(new_c) 

        ### get totfam @ time t
        totfam = new_fam * new_c 
        
        history_V_dep.append(new_fam)
        history_total.append(totfam)

        #get answer prob
        p_yes,p_no = answer_prob(beta,totfam)
        
        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_dep': history_V_dep,
                  'history_total':history_total,
                  'history_C': history_C,
                  'params':[alpha, sigma, beta, lamd_a],
                  'log_like': model_evidence,
                  'suppl': FP_rate}
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        return (model_evidence,data_store)


def VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV(params, old_Vfam_dep, old_c, action):
    
    # Data & Parameter
    alpha, sigma, beta, lamd_a = params[0], params[1], params[2], params[3],

    ### Get all data for current trial
    old_Vfam = old_Vfam_dep # previos stim familiarity
    old_c = old_c # duh

    ### Update view_dep familarity
    new_fam,vfam_PE = update_view_dependent(lamd_a,alpha,old_Vfam) # compute new stim familiarity
    
    # Update C Fam   
    new_c,c_PE = update_context(sigma,old_c,old_Vfam)       

    ### get totfam @ time t
    totfam = new_fam * new_c 
    
    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)
    if action == 1:
        return np.log(p_yes)
    if action == 0:
         return np.log(p_no)
    
   


# ################################### Control Model: VIEW_INDEPENDENT MODEL #############################################
#params = [alpha, beta, lamd_a]
#data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

def VIEW_INDEPENDENT(data, params):
    
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, verbose = data[0],data[1],data[2],data[3],data[4]
    alpha, beta, lamd_a = params[0], params[1], params[2],

    # get false positive answer rate
    FP_rate = FP_rate_independent(lamd_a,VPN_output,numb_prev_presentations)

    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([i for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_total = []
    history_answer = []
    model_evidence = 0
    vfam_tot = []
    
    ### inner loop through trials
    for trial in range(len(stim_IDs)):
        stim_ID,action,num_pres = stim_IDs[trial],VPN_output[trial],numb_prev_presentations[trial]
        ### Get predicted V_fam, C from last trial
        old_Vfam = history_V[stim_ID][-1]
        
        # Update VFam
        new_Vfam,vfam_PE = update_view_independent(lamd_a,alpha,old_Vfam)
        vfam_tot.append(new_Vfam)
        ### get totfam
        totfam = new_Vfam    
        history_total.append(totfam)
        ### Update C & V
        newfamlist = history_V[stim_ID].copy() + [new_Vfam]
        history_V[stim_ID] = newfamlist
    
        #get answer prob
        p_yes,p_no = answer_prob(beta,totfam)

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V': history_V,
                  'vfam_tot':vfam_tot,
                  'history_total':history_total,
                  'params':[alpha, beta, lamd_a],
                  'log_like': model_evidence,
                  'suppl': FP_rate}
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        return (model_evidence,data_store)     

def VIEW_INDEPENDENT_CV(params, old_Vfam, action):
    
    alpha, beta, lamd_a = params[0], params[1], params[2],
    
    ### Get predicted V_fam, C from last trial
    old_Vfam = old_Vfam
        
    # Update VFam
    new_Vfam,vfam_PE = update_view_independent(lamd_a,alpha,old_Vfam)
    ### get totfam
    totfam = new_Vfam    

    
    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)

    if action == 1:
        return np.log(p_yes)
    if action == 0:
        return np.log(p_no)
    


# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODEL #############################################  
#params = [alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep]
#data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]

def VIEW_INDEPENDENTxVIEW_DEPENDENT(data, params):
    alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[2],data[3],data[4],data[5]
    
    # Get init vals
    FP_rate_dep = FP_rate_dependent(lamd_a_dep,VPN_output,numb_prev_presentations)
    FP_rate_ind = FP_rate_independent(lamd_a_ind,VPN_output,numb_prev_presentations)
    # # Get unique amount of stimuli
    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)
    
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate_ind])
    history_answer = []
    history_total = []
    history_V_depend_L = []
    history_V_independ_L = []
    ### inner loop through trials
    for trial in range(len(stim_IDs)):
        stim_ID_depend, stim_ID_indipend = stim_IDs_perspective[trial],stim_IDs[trial]
        action,num_pres_depend, num_pres_independ = VPN_output[trial],numb_presentations[trial],numb_prev_presentations[trial]
        ### Get all data for current trial
        old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
        old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
        
        # update view-independent
        new_fam_ind,new_fam_ind_PE = update_view_independent(lamd_a_ind,alpha_ind,old_fam_indipend)
        history_V_independ_L.append(new_fam_ind)
        
        # update view-dependent
        new_fam_dep,new_fam_dep_PE = update_view_dependent(lamd_a_dep,alpha_dep,old_fam_depend) # compute new stim familiarity
        history_V_depend_L.append(new_fam_dep)   
        #protocol data
        new_fam_indipend_list = history_V_independ[stim_ID_indipend].copy() + [new_fam_ind] # append new stim familiarity to list
        new_fam_depend_list = history_V_depend[stim_ID_depend].copy() + [new_fam_dep]
        history_V_depend[stim_ID_depend] = new_fam_depend_list # replace list in dict
        history_V_independ[stim_ID_indipend] = new_fam_indipend_list


        ### get totfam @ time t
        totfam = new_fam_ind + new_fam_dep   
        history_total.append(totfam)
    
        #get answer prob
        p_yes,p_no = answer_prob(beta,totfam)

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_depend': history_V_depend,
                  'history_V_independ': history_V_independ,
                  'history_V_depend_L':history_V_depend_L,
                  'history_V_independ_L':history_V_independ_L,                  
                  'history_total': history_total,
                  'params': [alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep],
                  'log_like': model_evidence,
                  'suppl': (FP_rate_ind,FP_rate_dep)}
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        return (model_evidence, data_store)  

def VIEW_INDEPENDENTxVIEW_DEPENDENT_CV(params,old_fam_depend,old_fam_indipend, action):
    alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],

    ### Get all data for current trial
    old_fam_depend = old_fam_depend
    old_fam_indipend = old_fam_indipend
    
    # update view-independent
    new_fam_ind,new_fam_ind_PE = update_view_independent(lamd_a_ind,alpha_ind,old_fam_indipend)

    # update view-dependent
    new_fam_dep,new_fam_dep_PE = update_view_dependent(lamd_a_dep,alpha_dep,old_fam_depend) # compute new stim familiarity

    ### get totfam @ time t
    totfam = new_fam_ind + new_fam_dep   


    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)

    
    if action == 1:
        return np.log(p_yes)
    if action == 0:
        return np.log(p_no)
    ### inner loop through trials
    # for stim_ID_depend, stim_ID_indipend, action,num_pres_depend, num_pres_independ,trial in zip(stim_IDs_perspective,stim_IDs,VPN_output,numb_presentations,numb_prev_presentations,range(len(stim_IDs))):
       

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODELxCONTEXT MODEL #############################################
#params = [alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep]
#data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]

    
def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data, params):
    alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],params[5]
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose = data[0], data[1], data[2], data[3], data[4], data[5]
    
    unq_stim,numb_pres_dep = view_dep_suppl_dat(stim_IDs_perspective)
    #DEPENDENT get false positive answer rate
    FP_rate_dep = FP_rate_dependent(lamd_a_dep,VPN_output,numb_pres_dep)
    
    #INDEPENDENT false positive answer rate
    FP_rate_ind = FP_rate_independent(lamd_a_ind,VPN_output,numb_prev_presentations)  
    
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([i for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([i for i in range(1,25)], [FP_rate_ind])
    history_C = [0]

    history_total = []
    history_answer = []
    model_evidence = 0
    history_V_depend_L = []
    history_V_independ_L = []
   
    ### inner loop through trials
    for trial in range(len(stim_IDs)):
        stim_ID_depend, stim_ID_indipend = stim_IDs_perspective[trial],stim_IDs[trial]
        action = VPN_output[trial]
        ### Get all data for current trial
        
        old_fam_depend = history_V_depend[stim_ID_depend][-1] # previos stim familiarity
        old_fam_indipend = history_V_independ[stim_ID_indipend][-1]
        old_c = history_C[-1]
        
        # update view-independent
        new_fam_ind,new_fam_ind_PE = update_view_independent(lamd_a_ind,alpha_ind,old_fam_indipend)

        # update view-dependent
        new_fam_dep,new_fam_dep_PE = update_view_dependent(lamd_a_dep,alpha_dep,old_fam_depend) # compute new stim familiarity

        # Update C Fam  
        context_PE = (old_c - (old_fam_depend+old_fam_indipend))
        new_c = old_c - (sigma * context_PE)    
        
        history_V_depend_L.append(new_fam_dep)
        history_V_independ_L.append(new_fam_ind)
        history_C.append(new_c)
        
        ### get totfam @ time t
        totfam = (new_fam_ind + new_fam_dep) * new_c
        history_total.append(totfam)
    
        #get answer prob
        p_yes,p_no = answer_prob(beta,totfam)

        if action == 1:
            history_answer.append(p_yes)
        if action == 0:
            history_answer.append(p_no)
            
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_depend': history_V_depend,
                  'history_V_independ': history_V_independ,
                  'history_V_depend_L': history_V_depend_L, 
                  'history_V_independ_L': history_V_independ_L,
                  'history_C':history_C,
                  'history_total': history_total,
                  'params': [alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep],
                  'log_like': model_evidence,
                  'suppl':[FP_rate_ind,FP_rate_dep, numb_prev_presentations, VPN_output]}
    if verbose == False:
        return -1*model_evidence
    elif verbose == True:
        return (model_evidence, data_store)

def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV(params,old_fam_depend,old_fam_indipend,old_c, action):
    alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],params[5]

    # update view-independent
    new_fam_ind,new_fam_ind_PE = update_view_independent(lamd_a_ind,alpha_ind,old_fam_indipend)

    # update view-dependent
    new_fam_dep,new_fam_dep_PE = update_view_dependent(lamd_a_dep,alpha_dep,old_fam_depend) # compute new stim familiarity

    # Update C Fam  
    context_PE = (old_c - (old_fam_depend+old_fam_indipend))
    new_c = old_c - (sigma * context_PE)        
    ### get totfam @ time t
    totfam = (new_fam_ind + new_fam_dep) * new_c


    #get answer prob
    p_yes,p_no = answer_prob(beta,totfam)
    
    if action == 1:
        return np.log(p_yes)
    if action == 0:
        return np.log(p_no)



########################################## CV




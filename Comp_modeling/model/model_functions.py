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
    all_stim = [i for i in stim_IDs_perspective]
    unq_stim = np.unique(all_stim,return_counts = True)

    # count occurences of stimuli
    new_numb_presentations = dict.fromkeys(unq_stim[0], 0)
    numb_presentations = [] # append number of presentations per stim over course off experiment
    for i in all_stim:
        new_numb_presentations[i] +=1
        numb_presentations.append(new_numb_presentations[i])
    return unq_stim,numb_presentations

def bic(n_param, sample_size, raw_LL):
    bic = (n_param*np.log(sample_size))-(2*raw_LL)
    return bic

################################### Winning Model: INDIPENDENTxCONTEXT #############################################
# partial // Argumentabfolge functools partials 
# donald knut

#params [alpha, sigma, beta, lamd_a,]
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
# verbose

def VIEW_INDIPENDENTxCONTEXT(data,cv_trial,params):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    VPN_output = data[0].copy()
    new_ID = data[1]
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    verbose = data[5]
    
    # get false positive answer rate
    #print(VPN_output)
    FP_rate = FP_rate_independent(lamd_a,
                                  VPN_output,
                                  numb_prev_presentations)

    
    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
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

        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
            
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
                      'init_val':{'init_v': FP_rate,'init_c' : 0},
                      'history_V_dict':history_V}
        return (model_evidence,data_store)
 
    

def VIEW_INDIPENDENTxCONTEXT_gen(data,cv_trial,params):
    
    alpha = params[0]
    sigma = params[1]
    beta = params[2]
    lamd_a = params[3]
    
    VPN_output = data[0].copy()
    new_ID = data[1]
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    verbose = data[5]
    
    # get false positive answer rate
    #print(VPN_output)
    FP_rate = FP_rate_independent(lamd_a,
                                  VPN_output,
                                  numb_prev_presentations)

    synth_data_subj = []
    for i in range(100):
        ### dict for Trackkeeping of history
        history_V = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
        history_C = [0]
        history_C_pe = []
        history_V_ind = []
        history_total = []
        history_answer = []
        model_evidence = 0
        vfam_PE_list = []
        
        synth_dat = []
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
            
            pred_answer = np.random.binomial(1,p_yes)
            synth_dat.append(pred_answer)
            # if no CV

            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
    

        synth_data_subj.append(synth_dat)
    return synth_data_subj
    
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
        return (np.log(p_yes),totfam)
    if action == 0:
        return (np.log(p_no),totfam)


################################### Control Model: VIEW_DEPENDENT #############################################
#params = [alpha, beta, lamd_a]
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]


def VIEW_DEPENDENT(data,cv_trial, params):
    alpha, beta, lamd_a = params[0], params[1], params[2]
    VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[3],data[4],data[5]
    
    ''' Each stimulus presented from each of the 3 perspectives constitutes a new stimulus i.e. 24*3 Perspectives <= 72 possible stim IDs. BUT perspective is random - not necessary 72 unq stim 
    Stimulus context does not play a role.'''
    
    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)

    FP_rate_dep = FP_rate_dependent(lamd_a,VPN_output,numb_presentations)
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([str(i) for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R

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
        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
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
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]



def VIEW_DEPENDENTxCONTEXT_DEPENDENT(data,cv_trial, params):
    
    # Data & Parameter
    alpha, sigma, beta, lamd_a = params[0], params[1], params[2], params[3],
    VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[3],data[4],data[5]
    

    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)
   
    FP_rate = FP_rate_dependent(lamd_a,VPN_output,numb_presentations)
   
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([str(i) for i in unq_stim[0]], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
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
        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer': history_answer,
                  'history_V_dep': history_V_dep,
                  'history_V_dict': history_V_depend,
                  'history_total':history_total,
                  'history_C': history_C,
                  'params':[alpha, sigma, beta, lamd_a],
                  'log_like': model_evidence,
                  'VD_init': FP_rate,
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
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]


def VIEW_INDEPENDENT(data, cv_trial, params):
    
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, verbose = data[0],data[1],data[2],data[3],data[5]
    alpha, beta, lamd_a = params[0], params[1], params[2],

    # get false positive answer rate
    FP_rate = FP_rate_independent(lamd_a,VPN_output,numb_prev_presentations)

    ### dict for Trackkeeping of history
    history_V = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
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
        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
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
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]

def VIEW_INDEPENDENTxVIEW_DEPENDENT_gen(data, params):
    alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[2],data[3],data[4],data[5]
    
    # Get init vals
    FP_rate_dep = FP_rate_dependent(lamd_a_dep,VPN_output,numb_prev_presentations)
    FP_rate_ind = FP_rate_independent(lamd_a_ind,VPN_output,numb_prev_presentations)
    # # Get unique amount of stimuli
    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)
    
    init = False
    synth_data_subj = []
    for i in range(100):
        ### dict for Trackkeeping of history
        history_V_depend = dict.fromkeys([str(i) for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
        history_V_independ = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate_ind])
        history_answer = []
        history_total = []
        history_V_depend_L = []
        history_V_independ_L = []
        
        synth_dat = []
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
            
            pred_answer = np.random.binomial(1,p_yes)[0]
            synth_dat.append()
    
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        if init == False:
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
        elif init == True:
            synth_data_subj.append(synth_dat)
        init == True
    return (model_evidence, data_store,synth_data_subj)  


def VIEW_INDEPENDENTxVIEW_DEPENDENT(data, cv_trial, params):
    alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose = data[0],data[1],data[2],data[3],data[4],data[5]
    
    # Get init vals
    FP_rate_dep = FP_rate_dependent(lamd_a_dep,VPN_output,numb_prev_presentations)
    FP_rate_ind = FP_rate_independent(lamd_a_ind,VPN_output,numb_prev_presentations)
    # # Get unique amount of stimuli
    unq_stim,numb_presentations = view_dep_suppl_dat(stim_IDs_perspective)
    
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([str(i) for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate_ind])
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
        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
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
 

# ################################### Control Model: VIEW_INDEPENDENTxVIEW_DEPENDENT MODELxCONTEXT MODEL #############################################
#params = [alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep]
#data_ALL = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data,cv_trial, params):
    alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep = params[0], params[1],params[2],params[3],params[4],params[5]
    VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose = data[0], data[1], data[2], data[3], data[4], data[5]
    
    unq_stim,numb_pres_dep = view_dep_suppl_dat(stim_IDs_perspective)
    #DEPENDENT get false positive answer rate
    FP_rate_dep = FP_rate_dependent(lamd_a_dep,VPN_output,numb_pres_dep)
    
    #INDEPENDENT false positive answer rate
    FP_rate_ind = FP_rate_independent(lamd_a_ind,VPN_output,numb_prev_presentations)  
    
    ### dict for Trackkeeping of history
    history_V_depend = dict.fromkeys([str(i) for i in unq_stim[0]], [FP_rate_dep]) #set initial value of view_ind_fam as FP rate A&T pg.8 R
    history_V_independ = dict.fromkeys([str(i) for i in range(1,25)], [FP_rate_ind])
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

        # if no CV
        if cv_trial is None:
            if action == 1:
                history_answer.append(p_yes)
            if action == 0:
                history_answer.append(p_no)
        # if CV
        elif (cv_trial is not None):
            # if model not at specified holdout trial
            if (cv_trial != trial):
                if action == 1:
                    history_answer.append(p_yes)
                elif action == 0:
                    history_answer.append(p_no)
            # if model at specified holdout trial
            elif (cv_trial == trial): 
                None
            
    model_evidence = np.log(history_answer).sum()
    data_store = {'history_answer':         history_answer,
                  'history_V_depend':       history_V_depend,
                  'history_V_independ':     history_V_independ,
                  'history_V_depend_L':     history_V_depend_L, 
                  'history_V_independ_L':   history_V_independ_L,
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


def data_cv(data_vpn):
    from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT    

    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize   
    np.random.seed(1993)
    unique_id, data, lbfgs_epsilon, verbose_tot = data_vpn
    vpn = unique_id    
    
    curr_dat_VPN = data
    # get data
    stim_IDs_VI=    curr_dat_VPN[:,2].copy()     #stimulus IDs of winning model 
    new_ID=         curr_dat_VPN[:,3].copy()     #trials where new ID is introduced 
    n_prev_pres=    curr_dat_VPN[:,5].copy()     #number_of_prev_presentations
    stim_IDs_VD=    curr_dat_VPN[:,4].copy()     #view dependent
    VPN_output =    curr_dat_VPN[:,1].copy()     #VPN answers
    verbose =       False
    
    #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

    data_ALL = [VPN_output, 
                new_ID, 
                n_prev_pres, 
                stim_IDs_VI, 
                stim_IDs_VD, 
                verbose] 
    
    # cv_scores = {'VIEW_INDIPENDENTxCONTEXT'                 :[],
    #              'VIEW_DEPENDENT'                           :[],
    #              'VIEW_DEPENDENTxCONTEXT_DEPENDENT'         :[],
    #              'VIEW_INDEPENDENT'                         :[],
    #              'VIEW_INDEPENDENTxVIEW_DEPENDENT'          :[],
    #              'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'  :[]}
    cv_scores =   {'VIEW_INDIPENDENTxCONTEXT'                 :[],
                 'VIEW_DEPENDENT'                           :[],
                 'VIEW_DEPENDENTxCONTEXT_DEPENDENT'         :[],
                 'VIEW_INDEPENDENT'                         :[],
                 'VIEW_INDEPENDENTxVIEW_DEPENDENT'          :[],
                 'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'  :[]}
    for trial in range(len(VPN_output)):
        

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
                        
        data_verbose_debug = {}
        res_evidence = pd.DataFrame(index=models_names)
        trialwise_data = {}
        bf_log_group = pd.DataFrame()
                 

        data_ALL_debug = data_ALL.copy()
        data_ALL_debug[-1] = True 
        
        ########## Model Optim
        
        #print('VIEW_INDIPENDENTxCONTEXT')

        bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
        
        part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,trial) 
        res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        params_m_1 = res1[0]
        parameter_est['VIEW_INDIPENDENTxCONTEXT'] = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None,params_m_1)
        
        ##### VIEW_DEPENDENT
        #print('VIEW_DEPENDENT')
                
        bounds_M2 = [(0,1),(.1,20),(0,2)]
        
        part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,trial) 
        res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                      approx_grad = True,
                                      bounds = bounds_M2, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                      epsilon=epsilon_param)
        params_m_2 = res2[0]
        parameter_est['VIEW_DEPENDENT'] = res2[0]
        m_2 = VIEW_DEPENDENT(data_ALL_debug,None, params_m_2)
        
        
        ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
        #print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

        bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
    
        part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,trial) 
        res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                      approx_grad = True,
                                      bounds = bounds_M3, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                      epsilon=epsilon_param)
        params_m_3 = res3[0]
        parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'] = res3[0]
        m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_ALL_debug,None, params_m_3)
        
        
        ##### VIEW_INDEPENDENT
        #print('VIEW_INDEPENDENT')

        bounds_M4 = [(0,1),(.1,20),(0,2)]
    
        part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,trial) 
        res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                      approx_grad = True,
                                      bounds = bounds_M4, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                      epsilon=epsilon_param)
        
        params_m_4 = res4[0]
        parameter_est['VIEW_INDEPENDENT'] = res4[0]
        m_4 = VIEW_INDEPENDENT(data_ALL_debug,None, params_m_4)
        
        
        ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
        #print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
                
        bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,trial) 
        res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                      approx_grad = True,
                                      bounds = bounds_M5, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                      epsilon=epsilon_param)
        
        params_m_5 = res5[0]
        m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_ALL_debug, None, params_m_5)
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'] = res5[0]
        
        ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
        #print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')

    
        bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL,trial) 
        res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                      approx_grad = True,
                                      bounds = bounds_M6, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                      epsilon=epsilon_param)
        
        params_m_6 = res6[0]
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'] = res6[0]
        m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_ALL_debug,None, params_m_6)
        
        # if trial == 2:
        cv_scores['VIEW_INDIPENDENTxCONTEXT'].append(np.log(m_1[1]['history_answer'][trial]))
        cv_scores['VIEW_DEPENDENT'].append(np.log(m_2[1]['history_answer'][trial]))
        cv_scores['VIEW_DEPENDENTxCONTEXT_DEPENDENT'].append(np.log(m_3[1]['history_answer'][trial]))
        cv_scores['VIEW_INDEPENDENT'].append(np.log(m_4[1]['history_answer'][trial]))
        cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENT'].append(np.log(m_5[1]['history_answer'][trial]))
        cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'].append(np.log(m_6[1]['history_answer'][trial]))
        
        # cv_scores['VIEW_INDIPENDENTxCONTEXT'].append(m_1)
        # cv_scores['VIEW_DEPENDENT'].append(m_2)
        # cv_scores['VIEW_DEPENDENTxCONTEXT_DEPENDENT'].append(m_3)
        # cv_scores['VIEW_INDEPENDENT'].append(m_4)
        # cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENT'].append(m_5)
        # cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'].append(m_6)
    cv_scores['VIEW_INDIPENDENTxCONTEXT'] = np.array(cv_scores['VIEW_INDIPENDENTxCONTEXT']).sum()
    cv_scores['VIEW_DEPENDENT'] = np.array(cv_scores['VIEW_DEPENDENT']).sum()
    cv_scores['VIEW_DEPENDENTxCONTEXT_DEPENDENT'] = np.array(cv_scores['VIEW_DEPENDENTxCONTEXT_DEPENDENT']).sum()
    cv_scores['VIEW_INDEPENDENT'] = np.array(cv_scores['VIEW_INDEPENDENT']).sum()
    cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENT'] = np.array(cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENT']).sum()
    cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'] = np.array(cv_scores['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']).sum()
    
    
    
    return (unique_id,cv_scores)


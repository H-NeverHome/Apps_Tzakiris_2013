# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:43:24 2020

@author: de_hauk
"""



def fit_models(data, n_calls, n_rand_calls, n_jobs, init_points):
    from skopt import gp_minimize, utils, space
    import pandas as pd
    import itertools as it
    import numpy as np
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    ### get-time
    import datetime
    start = datetime.datetime.now()
    
    ### Get data 
    data = data
    
    ### Get opt
    n_calls = n_calls
    n_rand_start = n_rand_calls  
    n_jobs = n_jobs
    
    ### Get function
    
    from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #### get unique IDS
    sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
    sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]
    
    ## setup parameter dimensions   
    alpha_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
    sigma_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
    beta_raw = np.around(np.linspace(0.1, 19.9, num=200),decimals = 1)
    lamda_raw = np.around(np.linspace(0, 1.9, num=200),decimals = 2)
    lamda_raw_1 = np.around(np.linspace(0, 1.9, num=200),decimals = 2)
    alpha_raw_1 = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
    
    ## setup initial test-points
    if init_points ==True:
            
        alpha_raw_init = np.around(np.linspace(0, 0.9, num=5),decimals = 1)
        sigma_raw_init = np.around(np.linspace(0, 0.9, num=5),decimals = 1)
        beta_raw_init = np.around(np.linspace(0.1, 19.9, num=5),decimals = 1)
        lamda_raw_init = np.around(np.linspace(0, 1.9, num=5),decimals = 1)
        lamda_raw_1_init = np.around(np.linspace(0, 1.9, num=2),decimals = 1)
        alpha_raw_1_init = np.around(np.linspace(0, 0.9, num=2),decimals = 1)
    
    ## idiosyncratic param space
    coding = 'onehot'
    alpha_cat = space.Categorical(categories=alpha_raw,name='alpha_cat',transform = coding) # {0,1} rate at which familiarity was aquired
    sigma_cat  = space.Categorical(categories=sigma_raw,name='sigma_cat',transform = coding) # {0,1} context dependent learning rate
    beta_cat  = space.Categorical(categories=beta_raw,name='beta_cat',transform = coding) # {0,20} general disposition of VPS towards stochasticity of actions
    lamda_cat = space.Categorical(categories=lamda_raw,name='lamda_cat',transform = coding) # {0,1} maximum familiarity
    alpha_cat_1 = space.Categorical(categories=alpha_raw_1,name='alpha_cat_1',transform = coding) # {0,1} rate at which familiarity was aquired
    lamda_cat_1 = space.Categorical(categories=lamda_raw_1,name='lamda_cat_1',transform = coding) # {0,1} maximum familiarity

    ########### Define Loss Functions 
    
    ##### VIEW_INDIPENDENTxCONTEXT_DEMPENDENT
    
    # Dimensions
    import itertools as it
    
    dimensions_win_cat = [alpha_cat, sigma_cat, beta_cat, lamda_cat]
    
    if init_points == True:
        dimensions_win_cat_init = [i for i in it.product(alpha_raw_init,sigma_raw_init,beta_raw_init, lamda_raw_init)]
    else:
        dimensions_win_cat_init = None
    # Loss functions
    
    @utils.use_named_args(dimensions=dimensions_win_cat)
    def VIEW_INDIPENDENTxCONTEXT_optim_cat(alpha_cat, sigma_cat, beta_cat, lamda_cat):
        result = VIEW_INDIPENDENTxCONTEXT(alpha_cat, sigma_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, False)
        model_ev = result
        return -1*model_ev
    
    
    
    ##### View Dependent
    
    # dimensions
    dim_view_dep_cat = [alpha_cat, beta_cat, lamda_cat]
    
    if init_points ==True:
        dim_view_dep_cat_init =[i for i in it.product(alpha_raw_init,beta_raw_init, lamda_raw_init)]
    else:
        dim_view_dep_cat_init = None
    # Loss function
    
    '''Categorical Parameter Space'''
    #(alpha, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose)
    @utils.use_named_args(dimensions=dim_view_dep_cat)
    def VIEW_DEPENDENT_optim_cat(alpha_cat, beta_cat, lamda_cat):  
        result = VIEW_DEPENDENT(alpha_cat, beta_cat, lamda_cat, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, False)
        model_ev = result
        return -1*model_ev
    
    
    
    ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
    #dimensions
    
    dim_view_dep_context_cat = [alpha_cat, sigma_cat, beta_cat, lamda_cat]
    
    if init_points ==True:
        dim_view_dep_context_cat_init =[i for i in it.product(alpha_raw_init,sigma_raw_init,beta_raw_init, lamda_raw_init)]
    else:
        dim_view_dep_context_cat_init = None
    # Loss functions
    
    '''Categorical Parameter Space'''
    @utils.use_named_args(dimensions=dim_view_dep_context_cat)
    def VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat(alpha_cat, sigma_cat, beta_cat, lamda_cat):
        result = VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha_cat, sigma_cat, beta_cat, lamda_cat, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, False)
        model_ev = result
        return -1*model_ev
    
    
    ##### VIEW_INDEPENDENT
    
    #dimensions
    dim_view_dep_cat = [alpha_cat, beta_cat, lamda_cat] 
    
    if init_points ==True:
        dim_view_dep_cat_init = [i for i in it.product(alpha_raw_init,beta_raw_init, lamda_raw_init)]
    else:
        dim_view_dep_cat_init = None
    # Loss functions
    
    '''Categorical Parameter Space'''
    #alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose)
    
    @utils.use_named_args(dimensions=dim_view_dep_cat)
    def VIEW_INDEPENDENT_optim_cat(alpha_cat, beta_cat, lamda_cat):
        result = VIEW_INDEPENDENT(alpha_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, False)
        model_ev = result
        return -1*model_ev
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
    
    #dimensions
    dim_view_ind_view_dep_cat = [alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1] 
    
    if init_points == True:
        dim_view_ind_view_dep_cat_init =[i for i in it.product(alpha_raw_init,alpha_raw_1_init,sigma_raw_init,beta_raw_init, lamda_raw_init, lamda_raw_1_init)]
    else:
        dim_view_ind_view_dep_cat_init = None
    # Loss functions
    
    '''Categorical Parameter Space'''
    
    @utils.use_named_args(dimensions=dim_view_ind_view_dep_cat)
    def VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cat(alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1):
        result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
        model_ev = result
        return -1*model_ev
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #dimensions
    dim_view_ind_view_dep_context_cat = [alpha_cat, alpha_cat_1,sigma_cat, beta_cat, lamda_cat, lamda_cat_1]
    
    if init_points ==True:
        dim_view_ind_view_dep_context_cat_init =[i for i in it.product(alpha_raw_init,alpha_raw_1_init,sigma_raw_init,beta_raw_init, lamda_raw_init, lamda_raw_1_init)]
    else:
        dim_view_ind_view_dep_context_cat_init = None    
    # Loss functions
    
    '''Categorical Parameter Space'''
    #(alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose)
    @utils.use_named_args(dimensions=dim_view_ind_view_dep_context_cat)
    def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cat(alpha_cat, alpha_cat_1,sigma_cat, beta_cat, lamda_cat, lamda_cat_1):
        result = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha_cat, alpha_cat_1, sigma_cat, beta_cat, lamda_cat, lamda_cat_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
        model_ev = result
        return -1*model_ev
    
    ###############################################################################################################
    #function optim
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
    eta_time = 0
    noise = 1e-10
    skopt_verbose = False
    # Set Seed
    rnd_state = 1993

    #if init_points
    
    for i,j in zip(sample_answer_clms,sample_perspective_clms):
        print(i)
        print('eta_t-',eta_time)
        start_it = datetime.datetime.now()
        #func calls & rand starts
    
        # data import
        stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
        new_ID = data['new_IDs'] #trials where new ID is introduced 
        numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
        stim_IDs_perspective = data[str(j)] #view dependent
        VPN_output = data[str(i)] #VPN answers

    
        ##### Model Optim
        print('M_1')
        #VIEW_INDIPENDENTxCONTEXT_optim_cat
        res2a = gp_minimize(func=VIEW_INDIPENDENTxCONTEXT_optim_cat,
                    dimensions=dimensions_win_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dimensions_win_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
        
        
        print('M_2')
        #VIEW_DEPENDENT_optim_cat
        res2b = gp_minimize(func=VIEW_DEPENDENT_optim_cat,
                    dimensions=dim_view_dep_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dim_view_dep_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
        
        print('M_3')
        #VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat
        res2c = gp_minimize(func=VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat,
                    dimensions=dim_view_dep_context_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dim_view_dep_context_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
        
        print('M_4')
        #VIEW_INDEPENDENT_optim_cat
        res2d = gp_minimize(func=VIEW_INDEPENDENT_optim_cat,
                    dimensions=dim_view_dep_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dim_view_dep_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')   
        
        print('M_5')
        #VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cat
        res2e = gp_minimize(func=VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cat,
                    dimensions=dim_view_ind_view_dep_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dim_view_ind_view_dep_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
        
        print('M_6')    
        #VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cat
        res2f = gp_minimize(func=VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cat,
                    dimensions=dim_view_ind_view_dep_context_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise,
                    x0 = dim_view_ind_view_dep_context_cat_init,
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge') 
           
        res_data = [res2a,res2b,res2c,res2d,res2e,res2f]
        for model,model_res in zip(models_names,res_data):
            results_history_optim[model].append(model_res)
        now_time = datetime.datetime.now()    
        if i == sample_answer_clms[0]:
            eta_time = (now_time-start_it)*len(sample_answer_clms)
        else:
            new_it = now_time-start_it
            eta_time = eta_time - new_it
            
            
    # get time
    end = datetime.datetime.now()
    duration = end-start
    ### set up data_storage
    res_person = {'model_ev': pd.DataFrame(index = [i for i in results_history_optim['models']]),
                  'params':{'models':results_history_optim['models']},
                  'data': data,
                  'duration':duration}
    for person, person_n in zip(results_history_optim['IDs'],range(len(results_history_optim['IDs']))):
        vpn_id = person
        model_ev = []
        params = []
        for model in results_history_optim['models']:
            model_ev.append(results_history_optim[model][person_n]['fun'])
            params.append(results_history_optim[model][person_n]['x'])
        res_person['model_ev'][vpn_id] = model_ev
        res_person['params'][vpn_id] = params

    return res_person
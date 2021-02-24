# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:19:12 2020

@author: hauke
"""

###### TODO

'''
-Implement Rest of Models
-Check Optimization Routines
'''






folder_path_data = r'J:\main_results'

### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
from skopt import gp_minimize, utils, space
import pandas as pd
import numpy as np
import datetime
#import yabox as yb

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Import the DE implementations
#from yabox.algorithms import DE, PDE

### Get data 
data = pd.read_csv(folder_path_data + r'/final_proc_dat_labjansen.csv')

### Get function

from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT

#from tqdm import tqdm





#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]


## idiosyncratic param space
alpha_skl = space.Real(name='alpha', low=0, high=1) # {0,1} rate at which familiarity was aquired
sigma_skl  = space.Real(name='sigma', low=0, high=1) # {0,1} context dependent learning rate
beta_skl  = space.Real(name='beta', low=0.1, high=20) # {0,20} general disposition of VPS towards stochasticity of actions
lamda_skl  = space.Real(name='lamd_a', low=0, high=2) # {0,1} maximum familiarity

alpha_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
sigma_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
beta_raw = np.around(np.linspace(0.1, 19.9, num=200),decimals = 2)
lamda_raw = np.around(np.linspace(0, 1.9, num=200),decimals = 2)

alpha_cat = space.Categorical(categories=alpha_raw,name='alpha_cat',transform = 'identity') # {0,1} rate at which familiarity was aquired
sigma_cat  = space.Categorical(categories=sigma_raw,name='sigma_cat',transform = 'identity') # {0,1} context dependent learning rate
beta_cat  = space.Categorical(categories=beta_raw,name='beta_cat',transform = 'identity') # {0,20} general disposition of VPS towards stochasticity of actions
lamda_cat = space.Categorical(categories=lamda_raw,name='lamda_cat',transform = 'identity') # {0,1} maximum familiarity
alpha_cat_1 = space.Categorical(categories=alpha_raw,name='alpha_cat_1',transform = 'identity') # {0,1} rate at which familiarity was aquired
lamda_cat_1 = space.Categorical(categories=lamda_raw,name='lamda_cat_1',transform = 'identity') # {0,1} maximum familiarity


example = alpha_cat.rvs(20)

### Param Spaces for each model




########### Define Loss Functions 

##### VIEW_INDIPENDENTxCONTEXT_DEMPENDENT

# Dimensions

dimensions_win_cont = [alpha_skl, sigma_skl, beta_skl, lamda_skl]
dimensions_win_cat = [alpha_cat, sigma_cat, beta_cat, lamda_cat]

# Loss functions

@utils.use_named_args(dimensions=dimensions_win_cont)
def VIEW_INDIPENDENTxCONTEXT_optim_cont(alpha, sigma, beta, lamd_a):
    result = VIEW_INDIPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs)
    model_ev = result[0]
    return -1*model_ev
@utils.use_named_args(dimensions=dimensions_win_cat)
def VIEW_INDIPENDENTxCONTEXT_optim_cat(alpha_cat, sigma_cat, beta_cat, lamda_cat):
    result = VIEW_INDIPENDENTxCONTEXT(alpha_cat, sigma_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, False)
    model_ev = result
    return -1*model_ev



##### View Dependent

# dimensions
dim_view_dep_cat = [alpha_cat, beta_cat, lamda_cat]
dim_view_dep_cont = [alpha_skl, beta_skl, lamda_skl]

# Loss function

''' Continuous Parameter Space'''
@utils.use_named_args(dimensions=dim_view_dep_cont)
def VIEW_DEPENDENT_optim_cont(alpha_cat, beta_cat, lamda_cat):  
    result = VIEW_DEPENDENT(alpha_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective)
    model_ev = result[0]
    return -1*model_ev

'''Categorical Parameter Space'''
#(alpha, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose)
@utils.use_named_args(dimensions=dim_view_dep_cat)
def VIEW_DEPENDENT_optim_cat(alpha_cat, beta_cat, lamda_cat):  
    result = VIEW_DEPENDENT(alpha_cat, beta_cat, lamda_cat, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, False)
    model_ev = result
    return -1*model_ev



##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
#(alpha, sigma, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose)
#dimensions
dim_view_dep_context_cont = [alpha_skl, sigma_skl, beta_skl, lamda_skl] 
dim_view_dep_context_cat = [alpha_cat, sigma_cat, beta_cat, lamda_cat] 

# Loss functions

''' Continuous Parameter Space'''
@utils.use_named_args(dimensions=dim_view_dep_context_cont)
def VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cont(alpha, sigma, beta, lamd_a):
    result = VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, stim_IDs, stim_IDs_perspective)
    model_ev = result[0]
    return -1*model_ev

'''Categorical Parameter Space'''
@utils.use_named_args(dimensions=dim_view_dep_context_cat)
def VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat(alpha_cat, sigma_cat, beta_cat, lamda_cat):
    result = VIEW_DEPENDENTxCONTEXT_DEPENDENT(alpha_cat, sigma_cat, beta_cat, lamda_cat, VPN_output, new_ID, stim_IDs, stim_IDs_perspective, False)
    model_ev = result
    return -1*model_ev


##### VIEW_INDEPENDENT

#dimensions
dim_view_dep_cont = [alpha_skl, beta_skl, lamda_skl] 
dim_view_dep_cat = [alpha_cat, beta_cat, lamda_cat] 

# Loss functions

''' Continuous Parameter Space'''
@utils.use_named_args(dimensions=dim_view_dep_cont)
def VIEW_INDEPENDENT_optim_cont(alpha, beta, lamd_a):
    result = VIEW_INDEPENDENT(alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs, False)
    model_ev = result
    return -1*model_ev

'''Categorical Parameter Space'''
#alpha, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose)

@utils.use_named_args(dimensions=dim_view_dep_cat)
def VIEW_INDEPENDENT_optim_cat(alpha_cat, beta_cat, lamda_cat):
    result = VIEW_INDEPENDENT(alpha_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, False)
    model_ev = result
    return -1*model_ev

##### VIEW_INDEPENDENTxVIEW_DEPENDENT

#dimensions
dim_view_ind_view_dep_cont = [alpha_skl, alpha_skl, beta_skl, lamda_skl, lamda_skl] 
dim_view_ind_view_dep_cat = [alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1] 


# Loss functions

''' Continuous Parameter Space'''
@utils.use_named_args(dimensions=dim_view_ind_view_dep_cont)
def VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cont(alpha, alpha_1, beta, lamd_a, lamd_a_1):
    result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha, alpha_1, beta, lamd_a, lamd_a_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
    model_ev = result
    return -1*model_ev

'''Categorical Parameter Space'''
#(alpha_ind, alpha_dep, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose)

@utils.use_named_args(dimensions=dim_view_ind_view_dep_cat)
def VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cat(alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1):
    result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha_cat, alpha_cat_1, beta_cat, lamda_cat, lamda_cat_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
    model_ev = result
    return -1*model_ev

##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
#(alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose)
#dimensions
#dim_view_ind_view_dep_context_cont = [alpha_skl, alpha_skl,sigma_skl beta_skl, lamda_skl, lamda_skl] 
dim_view_ind_view_dep_context_cat = [alpha_cat, alpha_cat_1,sigma_cat, beta_cat, lamda_cat, lamda_cat_1] 


# Loss functions

# ''' Continuous Parameter Space'''
# @utils.use_named_args(dimensions=dim_view_ind_view_dep_cont)
# def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cont(alpha, alpha_1, beta, lamd_a, lamd_a_1):
#     result = VIEW_INDEPENDENTxVIEW_DEPENDENT(alpha, alpha_1, beta, lamd_a, lamd_a_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
#     model_ev = result
#     return -1*model_ev

'''Categorical Parameter Space'''
#(alpha_ind, alpha_dep, sigma, beta, lamd_a_ind, lamd_a_dep, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose)
@utils.use_named_args(dimensions=dim_view_ind_view_dep_context_cat)
def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cat(alpha_cat, alpha_cat_1,sigma_cat, beta_cat, lamda_cat, lamda_cat_1):
    result = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(alpha_cat, alpha_cat_1, sigma_cat, beta_cat, lamda_cat, lamda_cat_1, VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, False)
    model_ev = result
    return -1*model_ev

###############################################################################################################
#function optim
res = []

model_ev_VIEW_INDIPENDENTxCONTEXT_optim = 0
model_ev_VIEW_DEPENDENT_optim = 0
model_ev_VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim = 0
model_ev_VIEW_INDEPENDENT_optim = 0
model_ev_VIEW_INDEPENDENTxVIEW_DEPENDENT = 0
model_ev_VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim = 0
results_history_optim = []
n_calls = 20
n_rand_start = 10  
n_jobs = 18
noise = 1e-10
skopt_verbose = False
# Set Seed
rnd_state = 1993

for i,j in zip(sample_answer_clms[0:2],sample_perspective_clms[0:2]):
    print(i)
    #func calls & rand starts

    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    res_y0=[]

    ##### Model Optim
    
    res2a = gp_minimize(func=VIEW_INDIPENDENTxCONTEXT_optim_cat,
                    dimensions=dimensions_win_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise, 
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
    
    res2b = gp_minimize(func=VIEW_DEPENDENT_optim_cat,
                    dimensions=dim_view_dep_cat,
                    n_calls=n_calls,
                    random_state = rnd_state,
                    n_jobs=n_jobs,
                    n_random_starts = n_rand_start,
                    noise = noise, 
                    verbose = skopt_verbose,
                    acq_optimizer= 'sampling', 
                    acq_func = 'gp_hedge')
    
    res2c = gp_minimize(func=VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat,
                dimensions=dim_view_dep_context_cat,
                n_calls=n_calls,
                random_state = rnd_state,
                n_jobs=n_jobs,
                n_random_starts = n_rand_start,
                noise = noise, 
                verbose = skopt_verbose,
                acq_optimizer= 'sampling', 
                acq_func = 'gp_hedge')

    res2d = gp_minimize(func=VIEW_INDEPENDENT_optim_cat,
                dimensions=dim_view_dep_cat,
                n_calls=n_calls,
                random_state = rnd_state,
                n_jobs=n_jobs,
                n_random_starts = n_rand_start,
                noise = noise, 
                verbose = skopt_verbose,
                acq_optimizer= 'sampling', 
                acq_func = 'gp_hedge')    
    res2e = gp_minimize(func=VIEW_INDEPENDENTxVIEW_DEPENDENT_optim_cat,
                dimensions=dim_view_ind_view_dep_cat,
                n_calls=n_calls,
                random_state = rnd_state,
                n_jobs=n_jobs,
                n_random_starts = n_rand_start,
                noise = noise, 
                verbose = skopt_verbose,
                acq_optimizer= 'sampling', 
                acq_func = 'gp_hedge')        
    res2f = gp_minimize(func=VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim_cat,
                dimensions=dim_view_ind_view_dep_context_cat,
                n_calls=n_calls,
                random_state = rnd_state,
                n_jobs=n_jobs,
                n_random_starts = n_rand_start,
                noise = noise, 
                verbose = skopt_verbose,
                acq_optimizer= 'sampling', 
                acq_func = 'gp_hedge')        
       
    
    results_history_optim.append((i,('VIEW_INDIPENDENTxCONTEXT_optim_cat',res2a),
                                  ('VIEW_DEPENDENT_optim_cat', res2b),
                                  ('VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim_cat', res2c),
                                  ('VIEW_INDEPENDENT',res2d),
                                  ('VIEW_INDEPENDENTxVIEW_DEPENDENT',res2e),
                                  ('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',res2f)))
                                  

    # results_history_optim.append((i,('Bayesopt',res1),('Bayesopt_cat',res2),('Different Evo',de)))

    '''
    print('Model 2')
    res2 = gp_minimize(func= VIEW_DEPENDENT_optim,dimensions=dimensions_wo_context, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10  )
    print(res2['fun'], res2['x'])
    model_ev_VIEW_DEPENDENT_optim += res2['fun']
    
    print('Model 3')
    res3 = gp_minimize(func= VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim,dimensions=dimensions, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10  )
    print(res3['fun'], res3['x'])
    model_ev_VIEW_DEPENDENTxCONTEXT_DEPENDENT_optim += res3['fun']
    
    print('Model 4')
    res4 = gp_minimize(func= VIEW_INDEPENDENT_optim,dimensions=dimensions_wo_context, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10  )
    print(res4['fun'], res4['x'])
    model_ev_VIEW_INDEPENDENT_optim += res4['fun']
    
    print('Model 5')
    res5 = gp_minimize(func= VIEW_INDEPENDENTxVIEW_DEPENDENT_optim,dimensions=dimensions_wo_context, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10  )
    print(res5['fun'], res5['x'])
    model_ev_VIEW_INDEPENDENTxVIEW_DEPENDENT += res5['fun']
    
    print('Model 6')
    res6 = gp_minimize(func= VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim,dimensions=dimensions, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10  )
    print(res6['fun'], res6['x'])
    model_ev_VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_optim += res6['fun']
    
    results_history_optim.append((res1,res2,res3,res4,res5,res6))
    '''
'''
#################################################### Visualize ################################################

model_names = ['VIEW_INDIPENDENTxCONTEXT', 'VIEW_DEPENDENT', 'VIEW_DEPENDENTxCONTEXT_DEPENDENT', 'VIEW_INDEPENDENT', 'VIEW_INDEPENDENTxVIEW_DEPENDENT', 'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']    
vpn_viz = range(1,11)

All_dat_raw = pd.DataFrame()
All_dat_raw['models'] = pd.Series(model_names)
## put into dataframes
for i,j in zip(results_history_optim, range(len(results_history_optim))):
    Model_ev_VPN = []
    for model_ev in i:
        Model_ev_VPN.append(model_ev['fun'])
    All_dat_raw[str(j)] = pd.Series(Model_ev_VPN)
    
All_dat = All_dat_raw.set_index(keys = 'models')     

All_dat.to_csv(folder_path_WD + '/prelim_res.csv')
 
######################################### Todo  ########################################

#gradienten berechen -> autograd -> https://github.com/HIPS/autograd
# minimize(method=’L-BFGS-B’) -> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

### LBFGS Implement

    from scipy.optimize import minimize
    res = minimize(VIEW_INDIPENDENTxCONTEXT_optim_exp,
                                  method = 'L-BFGS-B',
                                  jac='2-point',
                                  x0 = [0.5,0.5,10,1],
                                  bounds = [(0.,1.),(0.,1.),(1.,20.),(0.,2.)],
                                  options = {'disp':True}
                                  )



'''
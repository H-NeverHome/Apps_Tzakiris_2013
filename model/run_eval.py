# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:19:12 2020

@author: hauke
"""



# new path D:\Apps_Tzakiris_rep\A_T_Implementation\data_lab_jansen
folder_path_data = r'E:\Opt old\main_results'
#folder_path_WD = r'D:\A_T_Implementation\impl_13_1_2019' # model_location
### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
from skopt import gp_minimize, utils, space
import pandas as pd
import numpy as np
import datetime


### Get data 
data = pd.read_csv(folder_path_data + r'/final_proc_dat_labjansen.csv')

### Get function

from model_functions import VIEW_INDIPENDENTxCONTEXT
from tqdm import tqdm


#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]


## idiosyncratic param space
alpha_skl = space.Real(name='alpha', low=0, high=1) # {0,1} rate at which familiarity was aquired
sigma_skl  = space.Real(name='sigma', low=0, high=1) # {0,1} context dependent learning rate
beta_skl  = space.Real(name='beta', low=0.1, high=20) # {0,20} general disposition of VPS towards stochasticity of actions
lamda_skl  = space.Real(name='lamd_a', low=0, high=2) # {0,1} maximum familiarity

alpha_raw = np.around(np.linspace(0.1, 0.9, num=5),decimals = 2)
sigma_raw = np.around(np.linspace(0.1, 0.9, num=5),decimals = 2)
beta_raw = np.around(np.linspace(0.1, 19.9, num=10),decimals = 2)
lamda_raw = np.around(np.linspace(0.1, 1.9, num=10),decimals = 2)
import itertools as it
    
res_space = [list(i) for i in it.product(alpha_raw,sigma_raw, beta_raw, lamda_raw)]


# alpha_skl = space.Categorical(categories = alpha_raw,name='alpha') # {0,1} rate at which familiarity was aquired
# sigma_skl  = space.Categorical(categories = sigma_raw,name='sigma') # {0,1} context dependent learning rate
# beta_skl  = space.Categorical(categories = beta_raw,name='beta') # {0,20} general disposition of VPS towards stochasticity of actions
# lamda_skl  = space.Categorical(categories = lamda_raw,name='lamd_a') # {0,1} maximum familiarity


### params for each model
dimensions = [alpha_skl, sigma_skl, beta_skl, lamda_skl]
dimensions_wo_context = [alpha_skl, beta_skl, lamda_skl]


### Define Loss Functions 

@utils.use_named_args(dimensions=dimensions)
def VIEW_INDIPENDENTxCONTEXT_optim(alpha, sigma, beta, lamd_a):
    result = VIEW_INDIPENDENTxCONTEXT(alpha, sigma, beta, lamd_a, VPN_output, new_ID, numb_prev_presentations, stim_IDs)
    model_ev = result[0]
    return -1*model_ev
#@utils.use_named_args(dimensions=dimensions)
def VIEW_INDIPENDENTxCONTEXT_optim_exp(x):
    result = VIEW_INDIPENDENTxCONTEXT(x[0], x[1], x[2], x[3], VPN_output, new_ID, numb_prev_presentations, stim_IDs)
    model_ev = result[0]
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
n_calls = 3
n_rand_start = 1

 # total model evidence
for i,j in zip(sample_answer_clms,sample_perspective_clms):
    print(i)
    #func calls & rand starts
    n_calls = 5100
    n_rand_start = 100
    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    res_y0=[]
    for params in tqdm(res_space):
        res_x0 = VIEW_INDIPENDENTxCONTEXT_optim_exp(params)
        res_y0.append(res_x0)
    ## time execution  

    ##optim
    #print(datetime.datetime.now().time())
    print('Model 1')
    res1 = gp_minimize(func=VIEW_INDIPENDENTxCONTEXT_optim,
                        dimensions=dimensions,
                        n_calls=n_calls,
                        x0= res_space,
                        y0 = res_y0,
                        n_jobs=8,
                        n_random_starts = 0,
                        noise =1e-10, 
                        verbose = True,
                        acq_optimizer= 'lbfgs', 
                        acq_func = 'gp_hedge')
    
    print(res1['fun'], res1['x'])
    model_ev_VIEW_INDIPENDENTxCONTEXT_optim += res1['fun']
    #print(datetime.datetime.now().time())
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
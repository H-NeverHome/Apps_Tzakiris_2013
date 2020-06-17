# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:19:12 2020

@author: hauke
"""




folder_path_data = r'D:\A_T_Implementation\data_lab_jansen'
folder_path_WD = r'D:\A_T_Implementation\impl_13_1_2019' # model_location
### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from skopt import gp_minimize, utils, space
import pandas as pd
import numpy as np
import os
os.chdir(folder_path_WD)
import A_P_Model as apm
model = apm.A_P_Model()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



### Get data 
data = pd.read_csv(folder_path_data + r'/final_proc_dat_labjansen.csv')

#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]


### idiosyncratic param space
alpha_skl = space.Real(name='alpha', low=0, high=1) # {0,1} rate at which familiarity was aquired
sigma_skl  = space.Real(name='sigma', low=0, high=1) # {0,1} context dependent learning rate
beta_skl  = space.Real(name='beta', low=0.1, high=20) # {0,20} general disposition of VPS towards stochasticity of actions
lamda_skl  = space.Real(name='lamd_a', low=0, high=2) # {0,1} maximum familiarity

### params for each model
dimensions = [alpha_skl, sigma_skl, beta_skl, lamda_skl]
dimensions_wo_context = [alpha_skl, beta_skl, lamda_skl]


###############################################################################################################
#function optim
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
    n_calls = 200
    n_rand_start = 100
    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    
    ### Disable depreciation warnings from sk-learn:: needs to be in the file that calls the functiond
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    ### time execution  
    import datetime
    # optim
    print(datetime.datetime.now().time())
    print('Model 1')
    res1 = gp_minimize(func= VIEW_INDIPENDENTxCONTEXT_optim,dimensions=dimensions, n_calls=n_calls,n_jobs=-1,n_random_starts = n_rand_start,noise =1e-10, verbose = True,acq_optimizer= 'lbfgs', acq_func = 'gp_hedge',n_restarts_optimizer = 10, n_points = 20000 )
    print(res1['fun'], res1['x'])
    model_ev_VIEW_INDIPENDENTxCONTEXT_optim += res1['fun']
    print(datetime.datetime.now().time())
    
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
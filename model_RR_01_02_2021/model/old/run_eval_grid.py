# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:18:00 2020

@author: de_hauk
"""





import pandas as pd
import numpy as np

from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT
from sklearn.model_selection import ParameterGrid

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'

param_grid = {'beta': np.linspace(0.1,20,200), 
              'max_fam' : np.arange(0.01,2,0.01),
              'view_ind_LR' : np.arange(0.01,1,0.01),
              'context_LR' : np.arange(0.01,1,0.01)}
                          
grid = ParameterGrid(param_grid)


import datetime
from functools import partial
import scipy
from scipy import optimize
np.random.seed(1993)
### Get data 
data = pd.read_csv(folder_path_data + r'/final_proc_dat_labjansen.csv')

#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]

epsilon_param = 0.001
x_0_bfgs = 0.5




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

res_evidence = pd.DataFrame(index=models_names)
res_max = []

for i,j in zip(sample_answer_clms[0:2],sample_perspective_clms[0:2]):
    print(i)
    #func calls & rand starts

    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    verbose = False
    
    
    ##### Model Optim
    
    
    print('VIEW_INDIPENDENTxCONTEXT')
    data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1)
    res_tot = []
    for param_dict in grid:
        
        params = [param_dict['view_ind_LR'], 
                  param_dict['context_LR'], 
                  param_dict['beta'], 
                  param_dict['max_fam']]
        res = part_func_M1(params)
        res_tot.append((res,params))
    res_max.append(res_tot)
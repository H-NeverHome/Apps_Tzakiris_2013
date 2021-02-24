# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:26:38 2020

@author: de_hauk
"""

###### TODO

'''
-Implement Rest of Models
-Check Optimization Routines
'''






folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'


from skopt import gp_minimize, utils, space
import pandas as pd
import numpy as np
import datetime
from functools import partial
import scipy
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

#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]


########### Define Loss Functions 

##### VIEW_INDIPENDENTxCONTEXT_DEMPENDENT

# Loss functions


def dummy(data,params):
    VPN_output = data[0]
    new_ID = data[1]
    numb_prev_presentations = data[2]
    stim_IDs = data[3]
    bool_var  = data[4]
    alpha_cat = params[0]
    sigma_cat = params[1]
    beta_cat = params[2]
    lamda_cat = params[3]
    
    res = VIEW_INDIPENDENTxCONTEXT(alpha_cat, sigma_cat, beta_cat, lamda_cat, VPN_output, new_ID, numb_prev_presentations, stim_IDs, bool_var)
    return -1*res



###############################################################################################################

n_calls = 20
n_rand_start = 15  
n_jobs = 16
noise = 1e-10
skopt_verbose = False
# Set Seed
rnd_state = 1993


resdat = []
for i,j in zip(sample_answer_clms,sample_perspective_clms):
    print(i)
    #func calls & rand starts

    # data import
    stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
    new_ID = data['new_IDs'] #trials where new ID is introduced 
    numb_prev_presentations = data.iloc[:,3] #number_of_prev_presentations
    stim_IDs_perspective = data[str(j)] #view dependent
    VPN_output = data[str(i)] #VPN answers
    bool_var = False
    res_y0=[]

    data_opt = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, bool_var]

    part_func = partial(dummy,data_opt)





    res2 = scipy.optimize.fmin_l_bfgs_b(part_func,
                                  approx_grad = True,
                                  bounds = [(.1,.9), (.1,.9),(.1,19.9), (.1,1.9)], 
                                  x0 = [.5,.1,10,1.5],
                                  epsilon=.1)
    resdat.append(res2)
       


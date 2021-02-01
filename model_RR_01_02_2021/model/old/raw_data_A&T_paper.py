# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:39:21 2020

@author: de_hauk
"""

import pandas as pd
import numpy as np
from scipy import special

########## Data Transcribed from paper

##### Model_selection
# Formulas from: 
#   Glover, S., Dixon, P. 
#   Likelihood ratios: A simple and flexible statistic for empirical psychologists. 
#   Psychonomic Bulletin & Review 11, 791–806 (2004). 
#   https://doi.org/10.3758/BF03196706

ll_raw = {'View-dependent': [-1590,3],
          'View-dependent_context-dependent': [-1510,4],
          'View-INdependent': [-1480,3],
          'View-independent_context-dependent': [-1425,4],
          'View-independent_View-dependent': [-1475,5],
          'View-independent_View-dependent_context-dependent': [-1430,6],
          }
ll_dat = pd.DataFrame.from_dict(data=ll_raw)
ll_dat['indx']=['ll','n_param']
ll_dat.set_index(keys='indx',inplace=True)

model_win = 'View-independent_context-dependent'
model_cntrl = [i for i in ll_dat if i != model_win]


ll_fin = pd.DataFrame(index = ['LR','LR_corr'])
lr_uncorr= []
lr_correct = []
n_size = 16


for i in model_cntrl:
    # Get Log-Likelihoods
    cntrl_ll = ll_dat[i][0]
    win_ll = ll_dat[model_win][0]
    
    # Get n_params
    win_nparam = ll_dat[model_win][1]
    cntrl_nparam = ll_dat[i][1]
    
    # uncorr LR
    lr_var = -2*(cntrl_ll-win_ll)
    # corr LL -> Glover & Dixon, 2004
    lr_corr = lr_var*((np.exp((cntrl_nparam-win_nparam)))**(np.log(n_size)/2))
    
    # POSITIVE VALUES FAVOR WINNING MODEL
    ll_fin[i] = [lr_var]+[lr_corr]

# Compare Multiple Models, lambda_mult     
lmbda_mult = -2*(special.logsumexp(np.array(ll_dat.T['ll'])-ll_dat[model_win][0]))    
    
def model_selection_AT():
    import pandas as pd
    import numpy as np
    from scipy import special
    ########## Data Transcribed from paper
    
    ##### Model_selection
    # Formulas from: 
    #   Glover, S., Dixon, P. 
    #   Likelihood ratios: A simple and flexible statistic for empirical psychologists. 
    #   Psychonomic Bulletin & Review 11, 791–806 (2004). 
    #   https://doi.org/10.3758/BF03196706
    
    ll_raw = {'View-dependent': [-1590,3],
              'View-dependent_context-dependent': [-1510,4],
              'View-INdependent': [-1480,3],
              'View-independent_context-dependent': [-1425,4],
              'View-independent_View-dependent': [-1475,5],
              'View-independent_View-dependent_context-dependent': [-1430,6],
              }
    ll_dat = pd.DataFrame.from_dict(data=ll_raw)
    ll_dat['indx']=['ll','n_param']
    ll_dat.set_index(keys='indx',inplace=True)
    
    model_win = 'View-independent_context-dependent'
    model_cntrl = [i for i in ll_dat if i != model_win]
    
    
    ll_fin = pd.DataFrame(index = ['LR','LR_corr'])
    n_size = 16
    
    
    for i in model_cntrl:
        # Get Log-Likelihoods
        cntrl_ll = ll_dat[i][0]
        win_ll = ll_dat[model_win][0]
        
        # Get n_params
        win_nparam = ll_dat[model_win][1]
        cntrl_nparam = ll_dat[i][1]
        
        # uncorr LR
        lr_var = -2*(cntrl_ll-win_ll)
        # corr LL -> Glover & Dixon, 2004
        lr_corr = lr_var*((np.exp((cntrl_nparam-win_nparam)))**(np.log(n_size)/2))
        
        # POSITIVE VALUES FAVOR WINNING MODEL
        ll_fin[i] = [lr_var]+[lr_corr]
    
    # Compare Multiple Models, lambda_mult     
    lmbda_mult = -2*(special.logsumexp(np.array(ll_dat.T['ll'])-ll_dat[model_win][0]))
    return {'AT_model_selection_results_nparams': ll_raw,
            'lambda_mult_model_selection':lmbda_mult }    
        
        
        
    
    
    
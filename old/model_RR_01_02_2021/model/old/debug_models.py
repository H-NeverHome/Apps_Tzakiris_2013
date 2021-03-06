# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:48:29 2020

@author: de_hauk
"""



from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT,VIEW_INDEPENDENT,VIEW_INDEPENDENTxVIEW_DEPENDENT,VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'

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


# data import

#### get unique IDS
sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]
vpn_answer = sample_answer_clms[0]
vpn_perspective = sample_perspective_clms[0]


stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
new_ID = data['new_IDs'] #trials where new ID is introduced 
numb_prev_presentations = data['number_of_prev_presentations_raw '] #number_of_prev_presentations
stim_IDs_perspective = data[vpn_perspective] #view dependent
VPN_output = data[vpn_answer] #VPN answers

alpha_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
sigma_raw = np.around(np.linspace(0, 0.9, num=100),decimals = 2)
beta_raw = np.around(np.linspace(0.1, 19.9, num=200),decimals = 2)
lamda_raw = np.around(np.linspace(0, 1.9, num=200),decimals = 2)


alpha_cat = space.Categorical(categories=alpha_raw,name='alpha_cat',transform = 'identity') # {0,1} rate at which familiarity was aquired
sigma_cat  = space.Categorical(categories=sigma_raw,name='sigma_cat',transform = 'identity') # {0,1} context dependent learning rate
beta_cat  = space.Categorical(categories=beta_raw,name='beta_cat',transform = 'identity') # {0,20} general disposition of VPS towards stochasticity of actions
lamda_cat = space.Categorical(categories=lamda_raw,name='lamda_cat',transform = 'identity') # {0,1} maximum familiarity

example = alpha_cat.rvs(1)[0]
beta_cat.rvs(1)[0]
lamda_cat.rvs(1)[0]

params_m_1 = [.38,.74,5.27,.64]
params_m_2 = [.02,.16,18.41,1.44]
params_m_3 = [.29,2.89,.62]
params_m_4 = [.38,1.19,.74]
params_m_5 = [.83,.13,.8,1.82,1.02]
params_m_6 = [.55,.02,.33,.5,1.07,1.49]

verbose = True

m_1 = VIEW_INDIPENDENTxCONTEXT([VPN_output, new_ID, numb_prev_presentations, stim_IDs, verbose],
                               [params_m_1[0],
                               params_m_1[1],
                               params_m_1[2],
                               params_m_1[3]] )

m_2 = VIEW_DEPENDENT([VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose],
                     [params_m_3[0], params_m_3[1], params_m_3[2]])

m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT([VPN_output, new_ID, stim_IDs, stim_IDs_perspective, True],
                                       [params_m_2[0],
                                       params_m_2[1],
                                       params_m_2[2],
                                       params_m_2[3]] )

m_4 = VIEW_INDEPENDENT([VPN_output, new_ID, numb_prev_presentations, stim_IDs,True],
                       [params_m_4[0],
                       params_m_4[1],
                       params_m_4[2]])

m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT([VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, True],
                                      [params_m_5[0], 
                                      params_m_5[1], 
                                      params_m_5[2],
                                      params_m_5[3],
                                      params_m_5[4]])

m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT([VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, True],[params_m_6[0],
                                              params_m_6[1],
                                              params_m_6[2],
                                              params_m_6[3],
                                              params_m_6[4],
                                              params_m_6[5]])

res_debug = [m_1,m_2,m_3,m_4,m_5,m_6]

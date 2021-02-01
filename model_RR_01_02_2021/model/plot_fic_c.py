# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:40:02 2020

@author: de_hauk
"""
import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT
from class_import import fit_data_noCV_irr_len_data,data_fit_t1_t2_comb
from class_import import bayes_RFX_cond,orig_procedure
from class_import import reformat_data_within_T,bic,fit_data_CV
from class_import import task_rel,corr_lr_func
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
import seaborn as sns


Fig_1_c_right = pd.read_csv(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\datagen\Fig_1_c_right\Fig_1_c_right.csv')['0']
Fig_1_c_left = pd.read_csv(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\datagen\Fig_1_c_left\transcribed_raw_FIG_1_c_L.csv')['0']
at_transc = pd.DataFrame()
at_transc['Fig_1_c_right'] = Fig_1_c_right
at_transc['Fig_1_c_left'] = Fig_1_c_left
########### Get Data

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
#data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

sns.set_context("paper", font_scale = 2.5, rc={"grid.linewidth": 1.,
                                             'lines.linewidth': 3.,}) 
sns.set_style("whitegrid")
fig, ax = plt.subplots(2,1,figsize=(6,6)) 
grnd_truth = data_2_sample['stat_ground-truth']                           
grnd_truth['Number of previous presentations (roll. avg. 5 trials)'] = grnd_truth['number_of_prev_presentations_raw '].rolling(window=5).mean().fillna(value=0.)
grnd_truth['Percentage of novel faces (roll. avg. 10 trials)'] = grnd_truth['new_IDs'].rolling(window=10,min_periods = 10).mean().fillna(value=0.)

##### Number of previous presentations (roll. avg. 5 trials)
n_prev = pd.DataFrame()
n_prev['Our approximation'] = grnd_truth['Number of previous presentations (roll. avg. 5 trials)']
n_prev['Fig. 1c left, Apps & Tsakiris (2013)'] = at_transc['Fig_1_c_left']

abc = sns.lineplot(data = n_prev, ax = ax[0])
abc.set_xticks(np.arange(0, 189, step=10), minor=False)
abc.set_yticks(np.arange(0, 13, step=1), minor=False)
ax[0].set_title('Number of previous presentations (roll. avg. 5 trials)')
##### Percentage of novel faces (roll. avg. 10 trials)
perc_nov = pd.DataFrame()
perc_nov['Our approximation'] = grnd_truth['Percentage of novel faces (roll. avg. 10 trials)']*100
perc_nov['Fig. 1c right, Apps & Tsakiris (2013)'] = [i*100 for i in at_transc['Fig_1_c_right']]

cba = sns.lineplot(data=perc_nov, ax = ax[1])
cba.set_xticks(np.arange(0, 189, step=10), minor=False)
cba.set_yticks(np.arange(0, 70, step=5), minor=False)
ax[1].set_title('Percentage of novel faces (roll. avg. 10 trials)')
ax[1].set_xlabel('Trials')
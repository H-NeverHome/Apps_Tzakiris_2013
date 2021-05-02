# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:34:56 2021

@author: de_hauk
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

save_path = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\res_power_analysis'
fig, axs = plt.subplots(1,2, sharey=True)

# # sns.lineplot(data=data_raw_fig1_c_l,ax = axs[0])
# # sns.lineplot(data=np.array(results_opt[-1][4]),ax = axs[0])
########## DOMINIC IDEA
pwr_analysis_D = pd.read_csv(save_path+'/' + 'fitted_data_power.csv', sep=',', index_col = 0)
sns.lineplot( x=[i for i in pwr_analysis_D.columns], y=pwr_analysis_D.T['ttest_arch_pwr'],ax = axs[0])
axs[0].axhline(y=0.9,color = 'red') 
axs[0].axvline(x=6,color = 'green') 
########## HAUKE IDEA 
pwr_analysis_H = pd.read_csv(save_path+'/' + 'synth_data_power.csv', sep=',', index_col = 0)
sns.lineplot( x=[i for i in pwr_analysis_H.columns], y=pwr_analysis_H.T['ttest_arch_pwr'],ax = axs[1])
axs[1].axhline(y=0.9,color = 'red')
axs[1].axvline(x=20,color = 'green')


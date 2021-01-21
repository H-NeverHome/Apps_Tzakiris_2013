# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:34:33 2021

@author: de_hauk
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\output'

data_T2 = pd.read_csv(path + '/fit_data_sample_T2.csv',index_col='Unnamed: 0')
data_T1 = pd.read_csv(path + '/fit_data_sample_T1.csv',index_col='Unnamed: 0')

data_T2_CV = pd.read_csv(path + '/fit_data_sample_T2_CV.csv',index_col='Unnamed: 0')
data_T1_CV = pd.read_csv(path + '/fit_data_sample_T1_CV.csv',index_col='Unnamed: 0')


sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": 1.,
                                             'lines.linewidth': 3.}) 
sns.set_style("whitegrid")
fig, ax = plt.subplots(2,1,figsize=(6,6)) 

data_merg = pd.merge(data_T1,data_T2,left_index=True, right_index=True)
data_merg_CV = pd.merge(data_T1_CV,data_T2_CV, left_index=True, right_index=True)

sns.barplot(data=data_merg.T, 
            palette = 'Blues',
            orient= 'h', 
            estimator = np.sum,
            ci = None,
            ax = ax[0])

sns.barplot(data=data_merg_CV.T, 
            palette = 'Blues',
            orient= 'h', 
            estimator = np.sum,
            ci = None,
            ax = ax[1])

ax[1].set_xlim(-800,-225)

ax[0].set_title('Raw LL')

ax[1].set_xlim(-800,-225)
ax[1].set_title('LL_LOOCV')
plt.tight_layout()


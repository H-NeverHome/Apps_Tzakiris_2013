# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:48:28 2020

@author: de_hauk
"""

from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

data_1_sample_raw = data_old_sample(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\A_T_Implementation\data_lab_jansen')



# fit data
fit_data = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)

# Plot Fig 2c
ids_vpn = [i for i in data_1_sample_raw['DATA_imput'] if 'answer' in i]
only_answ = data_1_sample_raw['DATA_imput'][ids_vpn+['number_of_prev_presentations_raw ']]
perc_yes = pd.DataFrame()
for i in np.unique(only_answ['number_of_prev_presentations_raw ']):
    aaa = only_answ.loc[only_answ['number_of_prev_presentations_raw '] == i][ids_vpn].mean(axis=0)
    perc_yes[str(i)] = aaa
    print(i)

###############################################################################
#       Modeling analyses in chronological order as in the document
###############################################################################

##### Model-Selection via t-Tests
model_selection_dat = fit_data['subj_evidence'].copy().T
res_model_selection = []
for cntrl_model in model_selection_dat[[i for i in model_selection_dat if i !='VIEW_INDIPENDENTxCONTEXT']]:
    cntr_data = model_selection_dat[cntrl_model]
    win_dat = model_selection_dat['VIEW_INDIPENDENTxCONTEXT']
    t_test = pg.ttest(win_dat, cntr_data, paired=False, tail='two-sided', correction='auto')
    res_model_selection.append(('VIEW_INDEPENDENTxCONTEXT' + ' vs. ' + cntrl_model,
                                t_test))

##### Predict Answers from V and C and compare to 0 w. t-test
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
res_tot = []
coeffs_c = []
coeffs_v = []
for vpn in ids_vpn:
    params_VD = fit_data['trialwise_dat'][vpn]
    iv_var = params_VD[['history_V', 'history_C']]
    dv = np.array(params_VD['vpn_answer'])
    
    ## Get unique Value of C
    #V -> C
    c_only = pg.logistic_regression(params_VD['history_V'],
                                    dv,
                                    alpha=0.05,
                                    as_dataframe=True, 
                                    remove_na=False)
     
    c_v = pg.logistic_regression(iv_var,
                                 dv,
                                 alpha=0.05,
                                 as_dataframe=True, 
                                 remove_na=False)
    debug = pg.logistic_regression(params_VD[['history_C','history_V']],
                                 dv,
                                 alpha=0.05,
                                 as_dataframe=True, 
                                 remove_na=False)
    coeffs_c.append(c_v['coef'].loc[2]) 
    coeffs_v.append(c_v['coef'].loc[1])
    # iv = params_VD[[i for i in params_VD if 'answer' not in i]]
    # 
    # logit_mod = sm.Logit(dv, iv)
    # logit_res = logit_mod.fit()
    #print(logit_res.summary())
    res_tot.append((c_only,c_v))
    
unique_C = pg.ttest(coeffs_c, 0, paired=False, tail='two-sided', correction='auto')
unique_V = pg.ttest(coeffs_v, 0, paired=False, tail='two-sided', correction='auto')
    
    
    
    # iv = params_VD[[i for i in params_VD if 'answer' not in i]]
    # dv = np.array(params_VD['vpn_answer'])
    # clf = LogisticRegression(random_state=0).fit(iv, dv) 
    # res = clf.coef_
    

# >>>   
    
# fit data w. cv?
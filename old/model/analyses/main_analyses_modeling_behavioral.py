# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:06:47 2020

@author: de_hauk
"""

from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 

# Data Path
folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'


data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

data_1_sample_raw = data_old_sample(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\A_T_Implementation\data_lab_jansen')

data_1_sample_raw = data_old_sample(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\A_T_Implementation\data_lab_jansen')
ids_vpn = [i for i in data_1_sample_raw['DATA_imput'] if 'answer' in i]


# Plot Fig 2c
ids_vpn = [i for i in data_1_sample_raw['DATA_imput'] if 'answer' in i]
only_answ = data_1_sample_raw['DATA_imput'][ids_vpn+['number_of_prev_presentations_raw ']]
perc_yes = pd.DataFrame()
for i in np.unique(only_answ['number_of_prev_presentations_raw ']):
    aaa = only_answ.loc[only_answ['number_of_prev_presentations_raw '] == i][ids_vpn].mean(axis=0)
    perc_yes[str(i)] = aaa
    print(i)




# fit data
fit_data = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)
fit_data_sample_2 = fit_data_noCV(data_2_sample, 0.01, False)
aaaa_v = fit_data_sample_2['subj_BF_log'].sum(axis=1)
aaaa_v2 = fit_data['subj_BF_log'].sum(axis=1)



###############################################################################
#       Behavioral analyses in chronological order as in the document
###############################################################################

##### Learning effects
data_lr = ids_vpn.copy() + [i for i in data_1_sample_raw['DATA_imput'] if 'presentations' in i]
lr_effects = data_1_sample_raw['DATA_imput'][data_lr].copy()
pres_123_raw = lr_effects.loc[lr_effects[data_lr[-1]] < 4]
pres_123= pres_123_raw[ids_vpn].sum()

pres_10_11_12_raw = lr_effects.loc[lr_effects[data_lr[-1]] >= 10]
pres_10_11_12 = pres_10_11_12_raw[ids_vpn].sum()

lr_res = pg.ttest(pres_123, pres_10_11_12, paired=True, tail='two-sided', correction='auto')

##### Consistency Effects @ t-1, t-2, t-3, t-10


#### BUGGED ####

consist_prev_t = [int(i) for i in range(1,11)]

data_consist = pd.DataFrame(index=consist_prev_t)
for vpn in ids_vpn:
    res_const_vpn = []
    for trial in consist_prev_t:
        vpn_dat = data_1_sample_raw['DATA_imput'][vpn]
        trials = len(vpn_dat)
        n_t_yes = len([i for i in vpn_dat if i==1])
        n_t_no = len([i for i in vpn_dat if i==0])
        consist_yes = 1e-4
        consist_no = 1e-4
        for n_trial, answer in enumerate(vpn_dat):
            if n_trial > trial:
                res = int(vpn_dat[n_trial]) + int(vpn_dat[n_trial-trial])
                if (res==2):
                    consist_yes+=1
                if (res==0):
                    consist_no+=1
            else:
                None
        #
        #print(n_t_no,consist_no)
        corr_consist_raw = (n_t_no/(consist_no)) +  (consist_yes/(n_t_yes))
        corr_consist = corr_consist_raw/trials
        res_const_vpn.append(corr_consist)
             
    data_consist[vpn] = res_const_vpn
    
data_consist_t = data_consist.copy().T

res_const_ttest_tot = pd.DataFrame(index = ['p_val', 't_val', 'cohens_d'])

for i in consist_prev_t[1::]:
    data_1 = data_consist_t[1]
    data_ttest = data_consist_t[i]
    res_const_ttest = pg.ttest(data_1, data_ttest, paired=True, tail='two-sided', correction='auto')
    res_const_rand = pg.ttest(data_1, 0, paired=False, tail='two-sided', correction='auto')
    res_const_ttest_tot['1 vs. chance??'] = [res_const_rand['p-val'][0], res_const_rand['T'][0], res_const_rand['cohen-d'][0]]
    res_const_ttest_tot['1 vs. ' + str(i)] = [res_const_ttest['p-val'][0], res_const_ttest['T'][0], res_const_ttest['cohen-d'][0]]

########## Supplemental Analyses  

         

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
    
    

    

##### Subject Specific Correlation with model parameters
data_corr = pd.DataFrame()
for vpn in ids_vpn:
    totfam = fit_data['trialwise_dat'][vpn]['history_total']
    answers = data_1_sample_raw['DATA_imput'][vpn]
    res_corr = pg.corr(totfam,answers, method = 'percbend')
    data_corr[vpn] = res_corr['r']
    
##### Predict Answers from V and C and compare to 0 w. t-test

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
    res_tot.append((c_only,c_v))
    
unique_C = pg.ttest(coeffs_c, 0, paired=False, tail='two-sided', correction='auto')
unique_V = pg.ttest(coeffs_v, 0, paired=False, tail='two-sided', correction='auto')

# fit data w. cv?
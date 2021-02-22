# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:53:51 2021

@author: de_hauk
"""

import os
os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\class')


from class_import import Apps_Tsakiris_2013

path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
path_ground_truth = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
pathtomodels = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\class'
data_analyses = Apps_Tsakiris_2013(path_data,
                                   path_ground_truth,
                                   pathtomodels)

data_123 = data_analyses.get_data(True)


#res = data_analyses.fit_data(False)

#res_beh = data_analyses.behavioral_performance()

import pandas as pd
data_raw = data_123
data_A = data_raw['A']
data_B = data_raw['B']
data_AB = {**data_A,**data_B}
unq_ids = list(data_AB.keys())

df_1_3_L_A = []
df_10_10_L_A = []
df_1_3_L_B = []
df_10_10_L_B = []
for vpn in unq_ids:
    curr_dat = data_AB[vpn]
    df_1_2_3 = curr_dat.loc[curr_dat['n_prev_VI'] <=3]['answer'].sum()/len(curr_dat.loc[curr_dat['n_prev_VI'] <=3]['answer'].index)
    df_10_11_12 = curr_dat.loc[curr_dat['n_prev_VI'] >=10]['answer'].sum()/len(curr_dat.loc[curr_dat['n_prev_VI'] >=10]['answer'].index)
    if 'A' in vpn:   
        df_1_3_L_A.append(df_1_2_3)
        df_10_10_L_A.append(df_10_11_12)
    elif 'B' in vpn:
        df_1_3_L_B.append(df_1_2_3)
        df_10_10_L_B.append(df_10_11_12)
        
p_answer = pd.DataFrame()
p_answer['A_1_3'] = df_1_3_L_A
p_answer['B_1_3'] = df_1_3_L_B
p_answer['A_10_12'] = df_10_10_L_A
p_answer['B_10_12'] = df_10_10_L_B
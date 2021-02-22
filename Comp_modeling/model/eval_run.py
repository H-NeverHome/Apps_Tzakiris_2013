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


res = data_analyses.fit_data_seperate(False)

res_RFX = data_analyses.RFX_modelselection(res)

res_beh = data_analyses.behavioral_performance()

res_123 = data_analyses.learning_effect()
      
res_relibility = data_analyses.task_reliability()

res_AT_model_select = data_analyses.model_selection_AT()
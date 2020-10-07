# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:48:28 2020

@author: de_hauk
"""

from class_import import get_data,get_data_2,data_old_sample, fit_data_noCV
from class_import import get_behavioral_performance,model_selection_AT
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import special 


########### Get Data

folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
#data_1_sample = get_data(folder_path_data)

data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                      r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')

data_1_sample_raw = data_old_sample(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\A_T_Implementation\data_lab_jansen')

########### Get Behavioral Performance
task_performance = get_behavioral_performance(data_2_sample)


########### Comp. Modeling

#get results from Apps&Tzakiris 2013

at_model = model_selection_AT()

# For intepretation see https://www.nicebread.de/a-short-taxonomy-of-bayes-factors/

# fit data sample N=10
fit_data_sample_1 = fit_data_noCV(data_1_sample_raw['DATA_imput'], 0.01, False)


# fit data sample N=3, T1 & T2
fit_data_sample_2 = fit_data_noCV(data_2_sample['imputed_data'], 0.01, False)






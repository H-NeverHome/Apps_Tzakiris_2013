# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:53:51 2021

@author: de_hauk
"""


########### Set WD to where class import file is located

'''
Set current WD where the modules are stored
Note, that the raw-data needs to be placed elsewhere!
'''
import os
path_to_class = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'

os.chdir(path_to_class)


########## Get class import
''' 
This class stores all relevant modules of the analyses mentioned in Zaragoza-Jimenez et al. 2021.
It needs to be in the same folder as all other files
'''
from class_import import Apps_Tsakiris_2013


########## Get list of installes packages and dump as list
import pkg_resources
import numpy as np
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
inst_pckg_arr = np.array(installed_packages_list)
np.savetxt('dependencies.txt',inst_pckg_arr, fmt='%s') 

##### location of data
path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'

##### location of ground truth-data
path_ground_truth = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
pathtomodels = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'
data_analyses = Apps_Tsakiris_2013(path_data,
                                   path_ground_truth,
                                   path_to_class)
########## preprocess & clean data
data_processed = data_analyses.get_data(verbose = True)

########## Behavioral analysis
res_behavioral = data_analyses.behavioral_performance()

res_learning_effect = data_analyses.learning_effect()


######### Reliability analysis
res_relibility = data_analyses.task_reliability()

########## Between-time model fitting
##### model selection results by apps & tsakiris 2013
res_AT_model_select = data_analyses.model_selection_AT()

# LOOCV score
#res_CV = data_analyses.fit_data_separate_LOOCV()

##### fit data seperate for each time-point
# Raw LL
res_fit_sep = data_analyses.fit_data_seperate(False)

# Corrected Likelihood ratio
res_LR = data_analyses.corr_LR()

# t-test procedure
res_ttest = data_analyses.ttest_procedure()

##### model selection results by apps & tsakiris 2013
res_AT_model_select = data_analyses.model_selection_AT()

########## Within-time model fitting
##### Model fitting under common set of MLE-params
res_fit_comb = data_analyses.fit_data_combined()

##### Random Effects within time analysis
res_RFX = data_analyses.RFX_modelselection(res_fit_sep)

##### time agn 10 LR
res_agn_10 = data_analyses.time_agn10_LR()
##########

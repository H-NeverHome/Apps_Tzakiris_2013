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
import pandas as pd

########## Get list of installes packages and dump as list
import pkg_resources
import numpy as np
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
inst_pckg_arr = np.array(installed_packages_list)
np.savetxt('dependencies.txt',inst_pckg_arr, fmt='%s') 

##### location of data
path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'

#### location of results
res_path = r'C:\Users\de_hauk\HESSENBOX\FRAPPS_RR (Nestor Israel Zaragoza Jimenez)\output_RR'

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

# save behavioral analysis
res_beh_path = res_path + '\\results_behavioral'
if not os.path.exists(res_beh_path):
    os.makedirs(res_beh_path)

curr_res_beh = res_behavioral['results']
curr_res_beh['M_res_AB'].to_html(res_beh_path + '\\mean_beh_performance_AB.html')
curr_res_beh['res_A'].to_html(res_beh_path + '\\mean_beh_performance_A.html')
curr_res_beh['res_B'].to_html(res_beh_path + '\\mean_beh_performance_B.html')

pd.DataFrame(res_learning_effect['AT_ES_D_poweranalysis'], index=[0]).to_html(res_beh_path + '\\AT_ES_D_poweranalysis.html')
res_learning_effect['answer']['descriptives'].to_html(res_beh_path + '\\stat_yes_asw.html')
res_learning_effect['perf']['descriptives'].to_html(res_beh_path + '\\stat_correct_yes_asw.html')
res_learning_effect['answer']['res_ttest'].to_html(res_beh_path + '\\learn_effect_yes_asw.html')
res_learning_effect['perf']['res_ttest'].to_html(res_beh_path + '\\learn_effect_correct_yes_asw.html')


######### Reliability analysis
res_reliability = data_analyses.task_reliability()

# save Reliability analysis
res_reliability_path = res_path + '\\reliability'
if not os.path.exists(res_reliability_path):
    os.makedirs(res_reliability_path)
    
res_reliability['corr'].to_html(res_reliability_path + '\\reliability_CORR.html')
res_reliability['icc'].to_html(res_reliability_path + '\\reliability_ICC.html')
        
########## Between-time model fitting/ seperate for each timepoint
# save model fit
res_fit_sep_path = res_path + '\\between_time_model_fit'
if not os.path.exists(res_fit_sep_path):
    os.makedirs(res_fit_sep_path)

##### model selection results by apps & tsakiris 2013
res_AT_model_select = data_analyses.model_selection_AT()

##### Fit models for each timepoint
res_fit_sep = data_analyses.fit_data_seperate(False)


pd.DataFrame(res_fit_sep['group_level_BIC_A']).to_html(res_fit_sep_path + '\\group_level_BIC_A.html')
pd.DataFrame(res_fit_sep['group_level_BIC_B']).to_html(res_fit_sep_path + '\\group_level_BIC_B.html')
pd.DataFrame(res_fit_sep['group_level_me_A']).to_html(res_fit_sep_path + '\\group_level_rawLL_A.html')
pd.DataFrame(res_fit_sep['group_level_me_B']).to_html(res_fit_sep_path + '\\group_level_rawLL_B.html')
pd.DataFrame(res_fit_sep['group_winmodel_post']).to_html(res_fit_sep_path + '\\group_winmodel_post.html')

##### Corrected Likelihood ratio on raw LL
res_LR = data_analyses.corr_LR()

pd.DataFrame(res_LR['corr_lr_A']).to_html(res_fit_sep_path + '\\corr_lr_A.html')
pd.DataFrame(res_LR['corr_lr_B']).to_html(res_fit_sep_path + '\\corr_lr_B.html')

##### t-test procedure on BIC
res_ttest = data_analyses.ttest_procedure()

res_ttest['t_test_A'].to_html(res_fit_sep_path + '\\t_test_A.html')
res_ttest['t_test_B'].to_html(res_fit_sep_path + '\\t_test_B.html')

##### LOOCV score
res_CV = data_analyses.fit_data_separate_LOOCV()
# # Save LOOCV score
# res_LOOCV_path = res_path + '\\LOOCV'
# if not os.path.exists(res_LOOCV_path):
#     os.makedirs(res_LOOCV_path)
    
pd.DataFrame(res_CV['subj_level_CV']).to_html(res_fit_sep_path + '\\subj_level_CV.html')

# ########## Within-time model fitting
# save model fit
within_time_modelfit = res_path + '\\within_time_modelfit'
if not os.path.exists(within_time_modelfit):
    os.makedirs(within_time_modelfit)
  
# ##### Model fitting under common set of MLE-params
res_fit_comb = data_analyses.fit_data_combined()


res_fit_comb.to_html(within_time_modelfit + '\\res_fit_comb.html')

# ##### time agn 10 LR on raw LL
res_agn_10 = data_analyses.time_agn10_LR()
res_agn_10['results'].to_html(within_time_modelfit + '\\res_agn_10.html')

###### Random Effects within time analysis on BIC
res_RFX = data_analyses.RFX_modelselection()
rfx_res = pd.DataFrame()
rfx_res[res_RFX[0]] = [res_RFX[1]]
rfx_res.to_html(within_time_modelfit + '\\rfx_res.html')


# ## Todo: save data to html!!!!
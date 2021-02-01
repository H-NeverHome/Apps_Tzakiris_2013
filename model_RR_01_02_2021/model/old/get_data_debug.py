# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:37:42 2020

@author: de_hauk
"""
import pickle
from get_data import data_old_sample
from run_eval_cat_debug import fit_models
from get_data_new_1 import data_new

data_old = data_old_sample()['DATA_imput']
data_new = data_new()
### fit_models(data, n_calls, n_rand_calls, n_jobs)

res_fit_new = fit_models(data_new, 500, 300, 16,False)

with open(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\meeting_13_08\res_fit_new1.pickle', 'wb') as handle:
    pickle.dump(res_fit_new, handle, protocol=pickle.HIGHEST_PROTOCOL)


res_fit_old = fit_models(data_old, 500, 300, 16,False)

with open(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\meeting_13_08\res_fit_old1.pickle', 'wb') as handle:
    pickle.dump(res_fit_old, handle, protocol=pickle.HIGHEST_PROTOCOL)



# model_ev = res_fit['model_ev']
# model_ev_1 = res_fit['model_ev']
# #model_ev['m_ev_sum'] = model_ev.sum(axis=1)

# sns.barplot(data=model_ev.T, palette="Blues")
# with open(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\meeting_13_08\res_fit_old1.pickle', 'rb') as handle:
#     b = pickle.load(handle)
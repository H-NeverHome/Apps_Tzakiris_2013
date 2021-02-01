
"""
Created on Mon Dec  7 14:17:14 2020

@author: de_hauk
"""
#hierarchical pymc3 model
if __name__ == "__main__":
    import pymc3 as pm
    import numpy as np
    from class_import import get_data,get_data_2
    import os
    from class_import import reformat_data_within_T
    os.chdir(r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\model')
    
    ########### Get Data
    
    folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    #data_1_sample = get_data(folder_path_data)
    
    data_2_sample = get_data_2(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv',
                          r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')
    ##### Reformat data for within-time format
    data_dict_t1 = reformat_data_within_T(data_2_sample)[1]
    data_dict_t2 = reformat_data_within_T(data_2_sample)[3]  
    
    data = np.array(data_dict_t1['1_A']['1_A_perf'])
    
    with pm.Model() as model:
        
        prior_s= pm.Uniform('prior_s',lower = 0, upper = 1)
        beta = pm.Exponential('beta',lam = prior_s)
        
        f_t = pm.Normal("f_t", mu=0, sigma=100)
               
        answer_pred = 1/(1+(pm.math.exp((-1*beta)*f_t)))
        observed = pm.Binomial('answ_prob',n=len(data),p=answer_pred,observed = data)
        #trace = pm.sample(2000, cores=1)
    

    with model:
        trace_mlr = pm.sample(1000, cores=4,tune = 5000,progressbar = True)
    
    #  if __name__ == "__main__":   
    res = pm.summary(trace_mlr)
    map_estimate = pm.find_MAP(model=model)
    # y = pm.Binomial.dist(n=10, p=0.5)
    yhf = model.logp

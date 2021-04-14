# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:16:24 2021

@author: de_hauk
"""


from tqdm import tqdm
import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize,special
np.random.seed(1993)
import numpy as np
import pandas as pd
import glob
import random 
from joblib import Parallel, delayed
import os
pathtomodels = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'

os.chdir(pathtomodels)

from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

from class_import_power_analysis import power_analyses


def fit_data_base(data_vpn):
    ### data has to be list
    
    from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    
    epsilon_param = .01
    x_0_bfgs = 0.5
    #func calls & rand starts
    vpn = data_vpn[0]
    curr_data_vpn = data_vpn
    # # get data numpy
    stim_IDs_VI=    curr_data_vpn[0:,2]  #stimulus IDs of winning model 
    new_ID=         curr_data_vpn[0:,1]      #trials where new ID is introduced 
    n_prev_pres=    curr_data_vpn[0:,4]    #number_of_prev_presentations
    stim_IDs_VD=    curr_data_vpn[0:,3]  #view dependent
    VPN_output =    curr_data_vpn[0:,0]     #VPN answers
    verbose =       False
    #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

    data_ALL = [VPN_output.astype(int), 
                new_ID.astype(int), 
                n_prev_pres.astype(int), 
                stim_IDs_VI.astype(str), 
                stim_IDs_VD.astype(str), 
                verbose]     
    ##### Data

    params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
    params_M2_name = ['alpha', 'beta', 'lamd_a']
    params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
    params_M4_name = ['alpha', 'beta', 'lamd_a']
    params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
    params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

    models_names = ['VIEW_INDIPENDENTxCONTEXT',
                    'VIEW_DEPENDENT',
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                    'VIEW_INDEPENDENT',
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
                    'random_choice']
    models_names_bic = ['VIEW_INDIPENDENTxCONTEXT',
                        'VIEW_DEPENDENT',
                        'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                        'VIEW_INDEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
    
    parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                    'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                    'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)}   

    
    
    ########## Model Optim
    
    
    print('VIEW_INDIPENDENTxCONTEXT')

    bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
    
    part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
    res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                  approx_grad = True,
                                  bounds = bounds_M1, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDIPENDENTxCONTEXT'][vpn] = res1[0]
    
    ##### VIEW_DEPENDENT
    print('VIEW_DEPENDENT')
            
    bounds_M2 = [(0,1),(.1,20),(0,2)]
    
    part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
    res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                  approx_grad = True,
                                  bounds = bounds_M2, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                  epsilon=epsilon_param)
    
    #parameter_est['VIEW_DEPENDENT'][vpn] = res2[0]
    
    ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
    print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

    bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]

    part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
    res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                  approx_grad = True,
                                  bounds = bounds_M3, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][vpn] = res3[0]
    
    ##### VIEW_INDEPENDENT
    print('VIEW_INDEPENDENT')

    bounds_M4 = [(0,1),(.1,20),(0,2)]

    part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
    res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                  approx_grad = True,
                                  bounds = bounds_M4, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDEPENDENT'][vpn] = res4[0]
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
    print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
            
    bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
    res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                  approx_grad = True,
                                  bounds = bounds_M5, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                  epsilon=epsilon_param)
    #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][vpn] = res5[0]
    
    ##### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
    #params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

    bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]

    part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL,None) 
    res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                  approx_grad = True,
                                  bounds = bounds_M6, 
                                  x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                  epsilon=epsilon_param)
    
    #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][vpn] = res6[0]
    
    
    ##### RND Choice model
    rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
    
    re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
    
    res_ev_DF = pd.DataFrame(index = models_names)
    res_ev_DF['ev'] = re_evidence_subj
    ### Subject BF_LOG
    bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj))
    
    return res_ev_DF


def ttest_procedure(subject_level_model_evidence):
    import pandas as pd
    import pingouin as pg
    import numpy as np
    ''' 
    As described in Apps & Tsakiris, we ran a series of independent
    two-sided t-tests and corrected via Benjamini–Hochberg false discovery
    rate (FDR)
    '''
    
    data_fit = subject_level_model_evidence.T
    
    win_model = 'VIEW_INDIPENDENTxCONTEXT'
    indx_models = [i for i in data_fit if 'VIEW_INDIPENDENTxCONTEXT' not in i]
    indx_res = ['p-val','cohen-d','BF10','obs_power','T']
    
    res = pd.DataFrame(index = indx_res)

    #for each time-point

    for model in indx_models:
        res_tt_raw = pg.ttest(data_fit[win_model],data_fit[model])
        res_tt_ind = [res_tt_raw['p-val'][0],
                      res_tt_raw['cohen-d'][0],
                      res_tt_raw['BF10'][0],
                      res_tt_raw['power'][0],
                      res_tt_raw['T'][0]]
        
        clm_name = win_model + '_vs_' + model
        res[clm_name] = res_tt_ind
    #FDR correction
    fdr_dat = res.copy().T.astype(float)
    fdr_dat['pval_FDR'] = pg.multicomp([i for i in fdr_dat['p-val']],
                                          alpha=0.02,
                                          method='fdr_bh')[1]
    return fdr_dat


import os
##### location of data
path_data = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
##### loc of class
path_to_class = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model'
### loc of ground truth
path_ground_truth = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'

### init class with paths
res_abba = power_analyses(path_data,path_ground_truth,path_to_class)

### get pilot_data
res_beh_power1 = res_abba.get_data(True)



########## DOMINIC IDEA 
'''use View-IndependentxContext model fitted with pilot data to generate
artificial data'''

### generate N datasets from each fitted model
res_beh_power2 = res_abba.gen_data(500)

res_123456 = {}
### for each sample size *3
for sample_size in tqdm([2,3,5,7,10,12,15]):
    print(sample_size*3)
    res_subsample = {}
    for subsample in range(30):
        print(('subsample',subsample))
        ###  only get data from first sample
        res_beh_power3 = []
        for key in [i for i in res_beh_power2.keys() if 'A' in i]:
            curr_dat = random.sample(res_beh_power2[key],sample_size)
            res_beh_power3 = res_beh_power3+curr_dat
        ### fit data and generate samples        
        res_gen_data1234 =  res_abba.fit_data_seperate(res_beh_power3,True)
        
        ###collect model evidence
        subj_lv_ev = pd.DataFrame(index = list(res_gen_data1234[0]['subject_level_model_evidence'].index))
        for indx,data_res in enumerate(res_gen_data1234):
            subj_data = list(data_res['subject_level_model_evidence']['ev'])
            subj_lv_ev[str(indx)] = subj_data
        #perform ttest procedure
        res_gen_data1234_TT = ttest_procedure(subj_lv_ev)
        post_prob = np.exp(np.sum(subj_lv_ev.iloc[0])-special.logsumexp(np.array(subj_lv_ev.iloc[0::].sum(axis=1))))
    
        #res_gen_data1234_pmp = np.exp(res_gen_data1234[1][0])/np.exp(special.logsumexp(list(res_gen_data1234[1])))
        res_subsample[str(subsample)] = {'ttest': res_gen_data1234_TT,
                                          'post_model_p':post_prob,
                                          'mean_power': (np.mean(res_gen_data1234_TT['obs_power']), res_gen_data1234_TT['obs_power'])}
    res_123456[str(sample_size*3)] = {'res_subsample':     res_subsample,
                               'raw_post_m_p':      [res_subsample[i]['post_model_p']for i in res_subsample],
                               'M_post_m_p':        np.mean([res_subsample[i]['post_model_p']for i in res_subsample]),
                               'raw_sample_power':  [res_subsample[i]['mean_power'][0] for i in res_subsample],
                               'mean_sample_power': np.mean([res_subsample[i]['mean_power'][0] for i in res_subsample])}

fitted_data_power = pd.DataFrame(index = ['ttest_arch_pwr', 'post_model_p'])
for sample_size in res_123456.keys():
    fitted_data_power[sample_size] = [np.around(res_123456[sample_size]['mean_sample_power'],decimals=5),
                                     np.around(res_123456[sample_size]['M_post_m_p'],decimals=5)]

save_path = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\res_power_analysis'
fitted_data_power.to_csv(save_path+'/' + 'fitted_data_power.csv', sep=',',index = True)

bbbaaa = pd.read_csv(save_path+'/' + 'fitted_data_power.csv', sep=',')


### get req_N for behavioral experiment
res_gen_data =  res_abba.behavioral_power()

########## HAUKE IDEA 
### generate synth data
'''Using Apps&Tsakiris figure to generate artificial samples of answer vectors'''

### generate 30 samples for each sample size between (min_sample_size, max_sample_size)
res_synth_data = res_abba.generate_data_pwr(10,60) 

### for every sample size
res_total_power = {}
for sample_size in tqdm(res_synth_data.keys()):
    print('curr_sample-size ' + str(sample_size))
    curr_ss = res_synth_data[sample_size]
    all_it = []
    ### convert every subsample of 30 to numpy
    for iteration in curr_ss.keys():
        curr_art_sample = curr_ss[iteration]
        curr_dat = []
        for j in curr_art_sample.keys():
            curr_dat.append(curr_art_sample[j].to_numpy())
        all_it.append(curr_dat)
        
    ### for every subsample/ iteration of sample
    res_total = []    
    for counter, ind_sample in enumerate(all_it):    
        print('subsample '+str(counter))
        # fit each generated subsample
        results123 = Parallel(n_jobs=-1,
                              verbose=0,
                              backend='loky')(delayed(fit_data_base)(k) for k in ind_sample)  
        
        #extract model_fit
        res_med = pd.DataFrame(index = list(results123[0].index))
        for counter, value in enumerate(results123):
            res_med[counter] = value
        #perform ttest-procedure
        res_tt = ttest_procedure(res_med)
        win_mod_ev = res_med.loc['VIEW_INDIPENDENTxCONTEXT'].sum()
        cntrl_mod = [res_med.loc[i].sum() for i in res_med.index]
        post_model_prob = np.exp(win_mod_ev)/np.exp(special.logsumexp(cntrl_mod))
        res_total.append({'subj_ev': res_med,
                          'res_ttest':res_tt,
                          'mean_obs_power' :    res_tt['obs_power'].mean(),
                          'post_model_prob' :   post_model_prob})
    
    res_total_power[sample_size] = {'tot_res':          res_total,
                                    'raw_post_m_p':     [i['post_model_prob'] for i in res_total],
                                    'M_post_m_p':       np.mean([i['post_model_prob'] for i in res_total]),
                                    'raw_sample_pwr':   [i['mean_obs_power'] for i in res_total],
                                    'M_sample_pwr':     np.mean([i['mean_obs_power'] for i in res_total])}
                                    


synth_data_power = pd.DataFrame(index = ['ttest_arch_pwr', 'post_model_p'])
for sample_size in res_total_power.keys():
    synth_data_power[sample_size] = [np.around(res_total_power[sample_size]['M_sample_pwr'] ,decimals=5),
                                      np.around(res_total_power[sample_size]['M_post_m_p']   ,decimals=5)]

save_path = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\res_power_analysis'
synth_data_power.to_csv(save_path+'/' + 'synth_data_power.csv', sep=',',index = True)

aaabbb = pd.read_csv(save_path+'/' + 'synth_data_power.csv', sep=',')

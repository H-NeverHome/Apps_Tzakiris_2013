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

def bic_data():
    params_M1_name = ['alpha', 'sigma', 'beta', 'lamd_a'] 
    params_M2_name = ['alpha', 'beta', 'lamd_a']
    params_M3_name = ['alpha', 'sigma', 'beta', 'lamd_a']
    params_M4_name = ['alpha', 'beta', 'lamd_a']
    params_M5_name = ['alpha_ind', 'alpha_dep', 'beta', 'lamd_a_ind', 'lamd_a_dep']
    params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']

    parameter_N = {'VIEW_INDIPENDENTxCONTEXT': len(params_M1_name),
                    'VIEW_DEPENDENT': len(params_M2_name),
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT': len(params_M3_name),
                    'VIEW_INDEPENDENT': len(params_M4_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT': len(params_M5_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': len(params_M6_name)}   
    return parameter_N

    
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
    
    data_fit = subject_level_model_evidence
    
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
                                          method='fdr_by')[1]
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
synth_data_gen = res_abba.gen_data(500)

### set params
n_subsamples = 30 # N of subsamples

samplesizes_it_raw = [i for i in range(2,11)]#+[15,20,25,30,35,40] # size of samples
samplesizes_it = samplesizes_it_raw[0::]

### for either generating mechanisms, view-independentXcontext, view-dependence
res_total_power = {}
vpn_keys = [i for i in synth_data_gen.keys()]
for gen_mech in vpn_keys:
    print(gen_mech)
    curr_gen_dat = synth_data_gen[gen_mech]
    
    ## for each timepoint
    res_total_timepoint = {}
    timpoints = ['A','B']
    for t in timpoints:
        
        # list of subjects
        keys = [i for i in curr_gen_dat.keys() if t in i]
        rawdata_subsample = []
        
        #convert data to long list
        for key in keys:
            rawdata_subsample = rawdata_subsample + [i for i in curr_gen_dat[key]]
        #res_123456 = {}
        res_total_samplesize = {}
        ### for each specified sample size
        for sample_size in tqdm(samplesizes_it):
            res_subsample = {}
            ### for each subsample of size sample_size
            for subsample in range(n_subsamples):
                print(('subsample',subsample))
                
                ### take random sample from generated data
                curr_dat_raw = random.sample(rawdata_subsample,sample_size)
                curr_dat = [i for i in curr_dat_raw]
                
                ### fit data and generate samples        
                res_gen_data1234,n_params =  res_abba.fit_data_seperate(curr_dat,True)
                subj_lv_ev = pd.DataFrame(index = list(res_gen_data1234[0]['subject_level_model_evidence'].index))
                for indx,data_res in enumerate(res_gen_data1234):
                    subj_data = list(data_res['subject_level_model_evidence']['ev'])
                    subj_lv_ev[str(indx)] = subj_data
                    
                ### compute BIC
                subj_lv_bic_raw = subj_lv_ev.copy().T
                subj_lv_bic_fin = pd.DataFrame()
                group_lv_bic_fin = pd.DataFrame()
                for model in n_params:
                    curr_modelev = list(subj_lv_bic_raw[model])
                    curr_nparams = n_params[model]
                    res_bic_model = []
                    #group_lv_bic_fin[model] = [bic(curr_nparams, sample_size, np.sum(curr_modelev))]
                    for subj in curr_modelev:
                        curr_bic = bic(curr_nparams, sample_size, subj)
                        res_bic_model.append(curr_bic)
                    subj_lv_bic_fin[model] = res_bic_model
                subj_lv_bic_fin['random_choice'] =  list(subj_lv_bic_raw['random_choice'])
                ### compute subjectwise post model probability of winning model
                m_prob = []   
                for indx in subj_lv_bic_raw.index:
                    subj_BIC_win = np.exp(float(subj_lv_bic_raw.loc[indx]['VIEW_INDIPENDENTxCONTEXT']))
                    subj_BIC_cntrl = np.exp(special.logsumexp(np.array(subj_lv_bic_raw.loc[indx])))
                    subj_post_prob = np.around(subj_BIC_win/subj_BIC_cntrl,decimals=5)
                    m_prob.append(subj_post_prob)
                comparison_params = []
                
                #perform ttest procedure
                tt_data = subj_lv_bic_fin.copy()
                res_gen_data1234_TT = ttest_procedure(tt_data)
                
                ### compute groupwise post model probability
                sum_BIC_win = float(subj_lv_bic_fin['VIEW_INDIPENDENTxCONTEXT'].sum())
                sum_BIC_cntrl = special.logsumexp(np.array(subj_lv_bic_fin.sum(axis=0)))
                post_prob = np.exp(np.around(sum_BIC_win-sum_BIC_cntrl,decimals=5))
        
        
                res_subsample[str(subsample)] = {'ttest':                   res_gen_data1234_TT,
                                                  'res_LL_subj':            subj_lv_ev,
                                                  'BIC_subj':               subj_lv_bic_fin,
                                                  'post_model_p_subj':      m_prob,
                                                  'post_model_p_subj_M':    np.around(np.mean(m_prob),decimals=5),
                                                  'post_model_p_group':     post_prob, 
                                                  'param_comp':             comparison_params,
                                                  'mean_power':             (np.mean(res_gen_data1234_TT['obs_power']), 
                                                                              res_gen_data1234_TT['obs_power'])}
                
                
            M_post_model_p_subj = np.mean([res_subsample[i]['post_model_p_subj_M']for i in res_subsample])
            SD_post_model_p_subj = np.std([res_subsample[i]['post_model_p_subj_M']for i in res_subsample])
        
            M_post_model_p_group = np.mean([res_subsample[i]['post_model_p_group']for i in res_subsample])
            SD_post_model_p_group = np.std([res_subsample[i]['post_model_p_group']for i in res_subsample])
               
            
            res_total_samplesize[str(sample_size)] = {
                                            'res_tt_subsample':                         res_subsample,
                                            'raw_post_model_p_subj':                    [res_subsample[i]['post_model_p_subj_M']for i in res_subsample],
                                            'M_post_model_p_subj':                      M_post_model_p_subj,
                                            'SD_post_model_p_subj':                     SD_post_model_p_subj,
                                            'model_p_subj_95%_CI':{'upper_bound':       M_post_model_p_subj+(1.645*SD_post_model_p_subj/np.sqrt(sample_size)),
                                                                   'lower_bound':       M_post_model_p_subj-(1.645*SD_post_model_p_subj/np.sqrt(sample_size)),
                                                                   },
                                            
                                            'M_post_model_p_group':                     M_post_model_p_group,
                                            'SD_post_model_p_group':                    SD_post_model_p_group,
                                            'model_p_group_95%_CI':{'upper_bound':      M_post_model_p_group+(1.645*SD_post_model_p_group/np.sqrt(sample_size)),
                                                                   'lower_bound':       M_post_model_p_group-(1.645*SD_post_model_p_group/np.sqrt(sample_size)),
                                                                   }, 
                                            
                                            'raw_post_model_p_group':                   [res_subsample[i]['post_model_p_group']for i in res_subsample],                                  
                                            'raw_sample_tt_power':                      [res_subsample[i]['mean_power'][0] for i in res_subsample],
                                            'mean_sample_tt_power':                     np.mean([res_subsample[i]['mean_power'][0] for i in res_subsample])}
            
            # if last iteration append summary
            if sample_size == samplesizes_it[-1]:
                summary = pd.DataFrame(index = [i for i in range(n_subsamples)])
                for ss in samplesizes_it:
                    summary[str(ss)] = res_total_samplesize[str(ss)]['raw_post_model_p_group']
                    summary = summary.round(5)
                res_total_samplesize['summ'] = summary
                
        res_total_timepoint[t] = res_total_samplesize

    res_total_power[gen_mech] = res_total_timepoint




# import seaborn as sns
# sns.set_theme(style="whitegrid")
# sns.set_context("paper")
# data = res_total_power['gen_view_ind_context']['A']['summ']
# # Draw a nested violinplot and split the violins for easier comparison
# sns.stripplot(data=data, palette = 'Blues')

# plt.ylim(.8, 1.1)






















































########## OLD ##########


#res12345_AA[curr_data] = res_123456
# # fitted_data_power = pd.DataFrame(index = ['ttest_arch_pwr', 'post_model_p'])
# # for sample_size in res_123456.keys():
# #     fitted_data_power[sample_size] = [np.around(res_123456[sample_size]['mean_sample_power'],decimals=5),
# #                                       np.around(res_123456[sample_size]['M_post_m_p'],decimals=5)]

# res_1234r = res_123456
# save_path = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\res_power_analysis'
# fitted_data_power.to_csv(save_path+'/' + 'fitted_data_power.csv', sep=',',index = True)

# bbbaaa = pd.read_csv(save_path+'/' + 'fitted_data_power.csv', sep=',')


# ### get req_N for behavioral experiment
# res_gen_data =  res_abba.behavioral_power()

# ########## HAUKE IDEA 
# ### generate synth data
# '''Using Apps&Tsakiris figure to generate artificial samples of answer vectors'''

# ### generate 30 samples for each sample size between (min_sample_size, max_sample_size)
# res_synth_data = res_abba.generate_data_pwr([25,30]) 
# parameter_N = bic_data()
# model_indx = [i for i in parameter_N]
# ### for every sample size
# res_total_power = {}
# for sample_size in tqdm(res_synth_data.keys()):
#     print('curr_sample-size ' + str(sample_size))
#     curr_ss = res_synth_data[sample_size]
#     all_it = []
#     ### convert every subsample of 30 to numpy
#     for iteration in curr_ss.keys():
#         curr_art_sample = curr_ss[iteration]
#         curr_dat = []
#         for j in curr_art_sample.keys():
#             curr_dat.append(curr_art_sample[j].to_numpy())
#         all_it.append(curr_dat)
        
#     ### for every subsample/ iteration of sample
#     res_total = []    
#     for counter, ind_sample in enumerate(all_it):    
#         print('subsample '+str(counter))
#         # fit each generated subsample
#         results123 = Parallel(n_jobs=-1,
#                               verbose=0,
#                               backend='loky')(delayed(fit_data_base)(k) for k in ind_sample)  
        
#         #extract model_fit
#         res_med = pd.DataFrame(index = list(results123[0].index))
#         for counter, value in enumerate(results123):
#             res_med[counter] = value
            
#         ### calc BIV
#         res_bic = pd.DataFrame()
#         for modell in model_indx:
#             raw_LL = res_med.loc[modell].sum()
#             curr_nparams = parameter_N[modell]
#             bic_proc = bic(curr_nparams, sample_size, raw_LL)
#             res_bic[modell] = [bic_proc]
#         #perform ttest-procedure
#         res_tt = ttest_procedure(res_med.T)
#         win_mod_ev = res_med.loc['VIEW_INDIPENDENTxCONTEXT'].sum()
#         cntrl_mod = [res_med.loc[i].sum() for i in res_med.index]
#         post_model_prob = np.exp(win_mod_ev)/np.exp(special.logsumexp(cntrl_mod))
#         res_total.append({'subj_ev': res_med,
#                           'res_ttest':res_tt,
#                           'mean_obs_power' :    res_tt['obs_power'].mean(),
#                           'post_model_prob' :   post_model_prob,
#                           'group_bic': res_bic})
    
#     res_total_power[sample_size] = {'tot_res':          res_total,
#                                     'raw_post_m_p':     [i['post_model_prob'] for i in res_total],
#                                     'M_post_m_p':       np.mean([i['post_model_prob'] for i in res_total]),
#                                     'raw_sample_pwr':   [i['mean_obs_power'] for i in res_total],
#                                     'M_sample_pwr':     np.mean([i['mean_obs_power'] for i in res_total])}
                                    


# synth_data_power = pd.DataFrame(index = ['ttest_arch_pwr', 'post_model_p'])
# for sample_size in res_total_power.keys():
#     synth_data_power[sample_size] = [np.around(res_total_power[sample_size]['M_sample_pwr'] ,decimals=5),
#                                       np.around(res_total_power[sample_size]['M_post_m_p']   ,decimals=5)]

# save_path = r'C:\Users\de_hauk\Documents\GitHub\Apps_Tzakiris_2013\Comp_modeling\model\res_power_analysis'
# synth_data_power.to_csv(save_path+'/' + 'synth_data_power.csv', sep=',',index = True)

# aaabbb = pd.read_csv(save_path+'/' + 'synth_data_power.csv', sep=',')

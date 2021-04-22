# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:55:02 2021

@author: de_hauk
"""

class power_analyses:
    def __init__(self,data_path,ground_truth_file,path_to_modelfunc):
        self.path_to_data = data_path
        self.path_to_groundtruth = ground_truth_file
        self.path_to_modelfunctions = path_to_modelfunc
        
    def get_data(self,verbose):
        ### get pilot_data
        import pandas as pd
        import glob
        import numpy as np
        all_files = glob.glob(self.path_to_data+ "/*.csv")
        
        SAMPLE_fullinfo = pd.read_csv(self.path_to_groundtruth).drop(columns = ['Unnamed: 0']).copy()  
        self.groundtruth = SAMPLE_fullinfo
        unq_id = []
        DATA_raw_DF = pd.DataFrame()
        
        final_dat_A = {}
        final_dat_B = {}
        for data in all_files:
            data_ID = pd.DataFrame()
            unique_ID = data[-12] + '_' + data[-5] 
            unq_id.append(unique_ID)
            curr_raw_data = pd.read_csv(data,header=None, sep='\t').drop(axis='index', labels = [0,1])[2]
            perspective = []
            answer_correct = [] #1 yes // 0 no
            answer_raw = [] # YES -> 1 // NO -> 0 // Do you remember face?
            for data_point in curr_raw_data:
                if len(data_point) < 5:
                    perspective.append(data_point)
                if ('1' in data_point[0])and (len(data_point)>5):
                    answer_correct.append(1)
                elif ('0' in data_point[0])and (len(data_point)>5):
                    answer_correct.append(0)
                if '[YES]' in data_point:
                    answer_raw.append(1)
                elif '[NO]' in data_point:
                    answer_raw.append(0)
                elif 'missed' in data_point:
                    answer_raw.append(np.nan)
            view_dep_stim_ID = {}       
            for VD_id,VD_num_ID in zip(np.unique(perspective),range(len(np.unique(perspective)))):
                view_dep_stim_ID[VD_id] = VD_num_ID
            view_dep_L = []
            for i in perspective:
                view_dep_L.append(view_dep_stim_ID[i])
            
            check_L = len(np.unique(perspective)) == len(np.unique(view_dep_L))
                
    
            data_ID['perf'] =           answer_correct
            data_ID['answer'] =         answer_raw
            data_ID['stim_IDs_VI'] =    SAMPLE_fullinfo['stim_IDs']
            data_ID['new_IDs'] =        SAMPLE_fullinfo['new_IDs']
            data_ID['stim_IDs_VD'] =    view_dep_L
            data_id_fin = data_ID.copy().loc[data_ID['answer'].isna()==False].reset_index(drop=True)
            
            n_prev_VI = []
            n_prev_VD = []
            unq_stim_ID_VI = {}
            unq_stim_ID_VD = {}
            for stim_VI,stim_VD in zip(data_id_fin['stim_IDs_VI'], data_id_fin['stim_IDs_VD']):
        
                key_VI, key_VD = str(stim_VI),str(stim_VD)
                
                ### count VI_Stims
                curr_keys_VI = [i for i in unq_stim_ID_VI.keys()]
                
                if key_VI not in curr_keys_VI:
                    unq_stim_ID_VI[key_VI] = 1
                elif key_VI in curr_keys_VI:
                    unq_stim_ID_VI[key_VI] = unq_stim_ID_VI[key_VI]+1
                
                n_prev_VI.append(int(unq_stim_ID_VI[key_VI]))
                
                ## count VD_Stims
                curr_keys_VD = [i for i in unq_stim_ID_VD.keys()]
                if key_VD not in curr_keys_VD:
                    unq_stim_ID_VD[key_VD] = 1
                elif key_VD in curr_keys_VD:
                    unq_stim_ID_VD[key_VD] = unq_stim_ID_VD[key_VD]+1
                n_prev_VD.append(int(unq_stim_ID_VD[key_VD]))  
                
            data_id_fin['n_prev_VI'] = n_prev_VI
            data_id_fin['n_prev_VD'] = n_prev_VD
            data_id_fin['stim_IDs_VI'] = [str(i) for i in data_id_fin['stim_IDs_VI']]
            data_id_fin['stim_IDs_VD'] = [str(i) for i in data_id_fin['stim_IDs_VD']]
            if 'A' in unique_ID:
                final_dat_A[unique_ID] = data_id_fin
            elif 'B' in unique_ID:
                final_dat_B[unique_ID] = data_id_fin
        final_dat = {'A':final_dat_A,
                     'B':final_dat_B
                     }
        self.clean_data = final_dat
        if verbose == True:
            return final_dat
        
        
    def behavioral_power(self):
        import pingouin as pg
        ''' Total Sample was 16, one excluded bc no learning effect'''

        ##### Table 6
        #T
        es_1 = pg.compute_effsize_from_t(14.66, N=15, eftype='cohen')
        req_N_1 = pg.power_ttest(d=es_1, n=None, power=0.9, alpha=0.02, contrast='one-sample', tail='two-sided')
        return {'AT_obs_effect_size': es_1,
                'AT_obs_p-val': 0.0001,
                'AT_N_to_replicate': req_N_1,
                'AT_obs_t-val':14.66,
                'AT_sample_N':15,
                'param_power':0.9,
                'param_alpha': 0.02}
    
    
    def generate_data_pwr(self, samples):
        path_ground_truth = self.path_to_groundtruth
        import numpy as np
        import pandas as pd
        import random
        at_sample_size = 15
        SAMPLE_fullinfo = pd.read_csv(path_ground_truth).drop(columns = ['Unnamed: 0'])
        prev_pres = np.array(SAMPLE_fullinfo[[i for i in SAMPLE_fullinfo][-1]])-1
        fig2_c_mean = [.2,
                       .43,
                       .58,
                       .7,
                       .74,
                       .76,
                       .83,
                       .77,
                       .78,
                       .81,
                       .82,
                       .78]
        fig2_c_SEM = [.09,
                      .08,
                      .07,
                      .09,
                      .09,
                      .08,
                      .09,
                      .1,
                      .05,
                      .07,
                      .07,
                      .09]
        fig2_c_SD_raw = np.array(fig2_c_SEM)
        fig2_c_SD = (fig2_c_SD_raw*np.sqrt(at_sample_size))/2
        def bounds(x):
            if x<=.0:
                return .0001
            elif x>=1.:
                return .9999
            else:
                return x
    
    
        res_beh_total = {}
        res_prob = {}
        for sample_size in samples:
            print(str(sample_size))
            res_beh = {}
            for sample in range(5):
                data_beh_synth = {}
                data_prob_synth = []
                for i in range(sample_size):
                    probs = []
                    perspective = []
                    for trial in prev_pres:
                        # generate probability
                        curr_prob_raw = np.random.normal(loc=fig2_c_mean[trial], scale=fig2_c_SD[trial])
                        # check bounds
                        curr_prob = bounds(curr_prob_raw)
                        probs.append(curr_prob)
                        #generate perspective data
                        persp = random.sample(['L','R','M'],1)[0]
                        perspective.append(persp)
                        
                    dat = np.random.binomial(1,probs)
                    trials_synth = pd.DataFrame()
                    trials_synth['answer'] =        dat
                    trials_synth['new_IDs'] =       SAMPLE_fullinfo['new_IDs']
                    trials_synth['stim_IDs_VI'] =   SAMPLE_fullinfo['stim_IDs']
                    trials_synth['perspective'] =   perspective
                    trials_synth['vdstim_raw'] = VD_stim = [str(i)+j for i,j in zip(trials_synth['stim_IDs_VI'],perspective)]
                    ##### VD count stim and prev_pres
                    unq_stim_VD = np.unique(VD_stim)
                    unq_stim_VI = np.unique(trials_synth['stim_IDs_VI'])
                    unq_stim_VD_dict = {}
                    #new numbering for VD stims
                    for key,val in zip(unq_stim_VD,range(len(unq_stim_VD))):
                        unq_stim_VD_dict[key] = val
                        
                    new_key_VD = [unq_stim_VD_dict[i] for i in VD_stim]
                    trials_synth['stim_IDs_VD'] = new_key_VD
                    ##### prev presentations
                    #VD
                    prev_pred_dict_VD = dict.fromkeys(np.unique(new_key_VD),0)
                    prev_pres_L_VD = []
                    #VI
                    prev_pred_dict_VI = dict.fromkeys(unq_stim_VI,0)
                    prev_pres_L_VI = []
                    
                    for vi,vd,trial in zip(list(trials_synth['stim_IDs_VI']),list(trials_synth['stim_IDs_VD']),range(len(list(trials_synth['stim_IDs_VD'])))):
                        prev_pred_dict_VI[vi] =  prev_pred_dict_VI[vi]+1
                        prev_pred_dict_VD[vd] =  prev_pred_dict_VD[vd] +1
                        prev_pres_L_VI.append(prev_pred_dict_VI[vi])
                        prev_pres_L_VD.append(prev_pred_dict_VD[vd])
                    
                    trials_synth['n_prev_VI'] =       prev_pres_L_VI
                    trials_synth['n_prev_VD'] =       prev_pres_L_VD
                    ###clean DF
                    trials_synth.drop(axis=1,labels = ['perspective','vdstim_raw'],inplace=True)
                    
                    ##save data
                    data_beh_synth[str(i)] = trials_synth
                    data_prob_synth.append(probs)
        
                res_beh[sample] = data_beh_synth
                #res_prob[sample_size] = data_prob_synth
            res_beh_total[sample_size] = res_beh
        return res_beh_total    
    
    def gen_data(self,synth_sample_N):
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic
        from model_functions import VIEW_INDIPENDENTxCONTEXT_gen,VIEW_DEPENDENT_gen
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)

        ### Get data 
        data_unform = self.clean_data 
        data_A = data_unform['A']
        data_B = data_unform['B']
        data_AB = {**data_A,**data_B}
        
        #### get unique IDS
        unique_id = list(data_AB.keys())

        
        epsilon_param = .01
        x_0_bfgs = 0.5
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
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                            }
                        
        data_verbose_debug = {}
        res_evidence = pd.DataFrame(index=models_names)
        trialwise_data = {}
        bf_log_group = pd.DataFrame()
        res_synth_tot = {}
        res_synth_tot1 = {}
        for vpn in unique_id:
            #func calls & rand starts
            print(vpn)

            curr_data_vpn = data_AB[vpn]

            # get data
            stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
            new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
            n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
            stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
            VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
            verbose =       False

            #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

            data_ALL = [VPN_output.astype(int), 
                        new_ID.astype(int), 
                        n_prev_pres.astype(int), 
                        stim_IDs_VI, 
                        stim_IDs_VD, 
                        verbose]            

            ########## Model Optim
            
            i=vpn
            print('VIEW_INDIPENDENTxCONTEXT')
    
            bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
            
            part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
            res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                          approx_grad = True,
                                          bounds = bounds_M1, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                          epsilon=epsilon_param)
            
            params_m_1 = res1[0]
            m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL,None,params_m_1)
            #synthetic_data_RND_param = VIEW_INDIPENDENTxCONTEXT_gen(data_ALL, synth_sample_N, params_m_1)         
            synthetic_data = VIEW_INDIPENDENTxCONTEXT_gen(data_ALL, synth_sample_N, params_m_1)
            synth_dat_DF = []
            for dat in synthetic_data:
                data_curr = pd.DataFrame()
                data_curr['answer'] = dat
                data_curr['stim_IDs_VI'] = stim_IDs_VI
                data_curr['new_IDs'] = new_ID
                data_curr['stim_IDs_VD'] = stim_IDs_VD
                data_curr['n_prev_VI'] = list(curr_data_vpn['n_prev_VI'])
                data_curr['n_prev_VD'] = list(curr_data_vpn['n_prev_VD'])
                synth_dat_DF.append(data_curr)
            
            res_synth_tot[vpn] = synth_dat_DF
            
            print('VIEW_DEPENDENT')
            bounds_M2 = [(0,1),(.1,20),(0,2)]
            
            part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
            res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                          approx_grad = True,
                                          bounds = bounds_M2, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                          epsilon=epsilon_param)

            params_m_2 = res2[0]
            m_2 = VIEW_DEPENDENT(data_ALL,None,params_m_1)
            synthetic_data2 = VIEW_DEPENDENT_gen(data_ALL, synth_sample_N, params_m_2)
            synth_dat_DF_2 = []
            for dat in synthetic_data2:
                data_curr = pd.DataFrame()
                data_curr['answer'] = dat
                data_curr['stim_IDs_VI'] = stim_IDs_VI
                data_curr['new_IDs'] = new_ID
                data_curr['stim_IDs_VD'] = stim_IDs_VD
                data_curr['n_prev_VI'] = list(curr_data_vpn['n_prev_VI'])
                data_curr['n_prev_VD'] = list(curr_data_vpn['n_prev_VD'])
                synth_dat_DF_2.append(data_curr)            
            
            res_synth_tot1[vpn] = synth_dat_DF_2
            
        self.synth_data = res_synth_tot
        return {'gen_view_ind_context': res_synth_tot,
                'gen_view_dep':         res_synth_tot1}

    def gen_data_rnd_param(self,synth_sample_N):
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT
        from model_functions import bic
        from model_functions import VIEW_INDIPENDENTxCONTEXT_gen
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)

        ### Get data 
        data_unform = self.clean_data 
        data_A = data_unform['A']
        data_B = data_unform['B']
        data_AB = {**data_A,**data_B}
        
        #### get unique IDS
        unique_id = list(data_AB.keys())

        
        epsilon_param = .01
        x_0_bfgs = 0.5
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
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                            }
                        
        data_verbose_debug = {}
        res_evidence = pd.DataFrame(index=models_names)
        trialwise_data = {}
        bf_log_group = pd.DataFrame()
        res_synth_tot = {}
        
        for vpn in unique_id:
            #func calls & rand starts
            print(vpn)

            curr_data_vpn = data_AB[vpn]

            # get data
            stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
            new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
            n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
            stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
            VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
            verbose =       False

            #data_All = [VPN_output, new_ID, numb_prev_presentations, stim_IDs_VI,stim_IDs_VD,verbose]

            data_ALL = [VPN_output.astype(int), 
                        new_ID.astype(int), 
                        n_prev_pres.astype(int), 
                        stim_IDs_VI, 
                        stim_IDs_VD, 
                        verbose]            

            ########## Model Optim
            
            i=vpn
            print('VIEW_INDIPENDENTxCONTEXT')
    
            bounds_M1 = [(.0,1),(.0,1),(.1,20),(.0,2)]

            synthetic_RND_param = []
            synthetic_data_RND_param = []
            for sample in range(synth_sample_N):
                rng1 = np.random.default_rng()
                rng2 = np.random.default_rng()
                rng3 = np.random.default_rng()
                rng4 = np.random.default_rng()
                aa = np.around(rng1.uniform(bounds_M1[0][0],bounds_M1[0][1],None),decimals=2)
                bb = np.around(rng2.uniform(bounds_M1[1][0],bounds_M1[1][1],None),decimals=2)
                cc = np.around(rng3.uniform(bounds_M1[2][0],bounds_M1[2][1],None),decimals=2)
                dd = np.around(rng4.uniform(bounds_M1[3][0],bounds_M1[3][1],None),decimals=2)
                rnd_params = (aa,bb,cc,dd)
                curr_dat = VIEW_INDIPENDENTxCONTEXT_gen(data_ALL, 1, rnd_params)
                synthetic_data_RND_param.append(curr_dat)
                synthetic_RND_param.append(rnd_params)
            synth_dat_DF = []
            for dat in synthetic_data_RND_param:
                data_curr = pd.DataFrame()
                data_curr['answer'] = dat[0]
                data_curr['stim_IDs_VI'] = stim_IDs_VI
                data_curr['new_IDs'] = new_ID
                data_curr['stim_IDs_VD'] = stim_IDs_VD
                data_curr['n_prev_VI'] = list(curr_data_vpn['n_prev_VI'])
                data_curr['n_prev_VD'] = list(curr_data_vpn['n_prev_VD'])
                synth_dat_DF.append(data_curr)
     
            
            
            res_synth_tot[vpn] = (synth_dat_DF,synthetic_RND_param)
        self.synth_data = res_synth_tot
        return res_synth_tot

    
    def gen_winmodel_power(self):
        None
    def gen_data_power(self):
        None

    def fit_data_seperate(self, data, verbose_tot):
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic
        from joblib import Parallel, delayed
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)

        def fit(data):
            epsilon_param = .01
            x_0_bfgs = 0.5
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
                            'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                                }
            
            data_ALL = data           

            ########## Model Optim
            
            print('VIEW_INDIPENDENTxCONTEXT')
    
            #bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
            bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
            part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL,None) 
            res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                          approx_grad = True,
                                          bounds = bounds_M1, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                          epsilon=epsilon_param,
                                          m=100)
            parameter_est['VIEW_INDIPENDENTxCONTEXT'] = res1[0]
            
            ##### VIEW_DEPENDENT
            print('VIEW_DEPENDENT')
                    
            bounds_M2 = [(0,1),(.1,20),(0,2)]
            
            part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
            res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                          approx_grad = True,
                                          bounds = bounds_M2, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                          epsilon=epsilon_param)
            
            parameter_est['VIEW_DEPENDENT'] = res2[0]
            
            ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
            print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

            bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
        
            part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
            res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                          approx_grad = True,
                                          bounds = bounds_M3, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'] = res3[0]
            
            ##### VIEW_INDEPENDENT
            print('VIEW_INDEPENDENT')
    
            bounds_M4 = [(0,1),(.1,20),(0,2)]
        
            part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
            res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                          approx_grad = True,
                                          bounds = bounds_M4, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENT'] = res4[0]
            
            ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
            print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
                    
            bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
            res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                          approx_grad = True,
                                          bounds = bounds_M5, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'] = res5[0]
            
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
            
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'] = res6[0]
            
            
            ##### RND Choice model
            rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
            
            re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
            pes_ev_DF = pd.DataFrame(index=models_names)
            pes_ev_DF['ev'] = re_evidence_subj
            
            ### Subject BF_LOG
            post_prob = np.exp(re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj)))

            ### Compute Bic

    
            
            ############################## Verbose == True ###########################
            
            verbose_debug = verbose_tot
            
            data_ALL_debug = data_ALL
            data_ALL_debug[-1] = True        
            
            if verbose_debug == True:
    
                # = data_M_debug[0]
                params_m_1 = res1[0]
                m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None,params_m_1)
                
                #data_M2_debug = data_M_debug[1]
                params_m_2 = res2[0]
                m_2 = VIEW_DEPENDENT(data_ALL_debug,None, params_m_2)
            
                #data_M3_debug = data_M_debug[2]
                params_m_3 = res3[0]
                m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_ALL_debug,None, params_m_3)
                
                #ädata_M4_debug = data_M_debug[3]
                params_m_4 = res4[0]
                m_4 = VIEW_INDEPENDENT(data_ALL_debug,None, params_m_4)
            
                #data_M5_debug = data_M_debug[4]
                params_m_5 = res5[0]
                m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_ALL_debug,None, params_m_5)
                
                #data_M6_debug = data_M_debug[5]
                params_m_6 = res6[0]
                m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_ALL_debug,None, params_m_6)
        
                res_debug = {models_names[0]: m_1,
                             models_names[1]: m_2,
                             models_names[2]: m_3,
                             models_names[3]: m_4,
                             models_names[4]: m_5,
                             models_names[5]: m_6}
                #data_verbose_debug[i] = res_debug
                
        #### Get winning model trialwise dat ####
            params_m_1 = res1[0]
            res_M1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None, params_m_1)        
    
            #trialwise_data[i] = res_M1[1]['data_store_1']
        
            results_1 = {'subject_level_model_evidence':pes_ev_DF,
                        'used_data': 'comp too expensive',
                        'post_prob': post_prob,
                        'subject_level_parameter-estimates':parameter_est}

            return results_1
       
        
        #### get unique IDS
        unique_id = [i for i in range(len(data))]

        
        # epsilon_param = .01
        # x_0_bfgs = 0.5
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
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                            }
        n_params = {'VIEW_INDIPENDENTxCONTEXT':                 len(params_M1_name),
                    'VIEW_DEPENDENT':                           len(params_M2_name),
                    'VIEW_DEPENDENTxCONTEXT_DEPENDENT':         len(params_M3_name),
                    'VIEW_INDEPENDENT':                         len(params_M4_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENT':          len(params_M5_name),
                    'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT':  len(params_M6_name)}
                        
        data_verbose_debug = {}
        res_evidence = pd.DataFrame(index=models_names)
        trialwise_data = {}
        bf_log_group = pd.DataFrame()
        
        data_multp = []
        for data1 in data:
            # get data
            stim_IDs_VI=    np.array(data1['stim_IDs_VI'])  #stimulus IDs of winning model 
            new_ID=         np.array(data1['new_IDs'])      #trials where new ID is introduced 
            n_prev_pres=    np.array(data1['n_prev_VI'])    #number_of_prev_presentations
            stim_IDs_VD=    np.array(data1['stim_IDs_VD'])  #view dependent
            VPN_output =    np.array(data1['answer'])       #VPN answers
            verbose =       False
            
            data_ALL = [VPN_output.astype(int), 
                        new_ID.astype(int), 
                        n_prev_pres.astype(int), 
                        stim_IDs_VI, 
                        stim_IDs_VD, 
                        verbose]     
            data_multp.append(data_ALL)
        
        results123 = Parallel(n_jobs=-1,
                              verbose=0,
                              backend='loky')(delayed(fit)(k) for k in data_multp) 
        return [results123,n_params]
        
           
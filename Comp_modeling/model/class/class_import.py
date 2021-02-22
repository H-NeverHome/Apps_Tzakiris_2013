# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:10:21 2021

@author: de_hauk
"""



class Apps_Tsakiris_2013:
    def __init__(self,data_path, ground_truth_file,path_to_modelfunc):
        self.path_to_data = data_path
        self.path_to_groundtruth = ground_truth_file
        self.path_to_modelfunctions = path_to_modelfunc
        
    def get_data(self,verbose):
        
        import pandas as pd
        import glob
        import numpy as np
        from autoimpute.imputations import MultipleImputer
        # data_path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
        # ground_truth_file = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
        
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
            for stim_VI,stim_VD in zip(data_id_fin['stim_IDs_VI'], data_id_fin['stim_IDs_VI']):
        
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

    def fit_data(self, verbose_tot):
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
        
        
        
        
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

            data_ALL = [VPN_output.astype(int), 
                        new_ID.astype(int), 
                        n_prev_pres.astype(int), 
                        stim_IDs_VI, 
                        stim_IDs_VD, 
                        verbose]            

            
            
            ##### Model Optim
            
            i=vpn
            print('VIEW_INDIPENDENTxCONTEXT')
            #data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
            bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
            
            part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_ALL) 
            res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                          approx_grad = True,
                                          bounds = bounds_M1, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
            
            
            print('VIEW_DEPENDENT')
            
            #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
            #data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
    
        
            bounds_M2 = [(0,1),(.1,20),(0,2)]
            
            part_func_M2 = partial(VIEW_DEPENDENT,data_ALL) 
            res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                          approx_grad = True,
                                          bounds = bounds_M2, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                          epsilon=epsilon_param)
            
            parameter_est['VIEW_DEPENDENT'][i] = res2[0]
            
            
            print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')
    
            #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
            
            #data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
            bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
        
            part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL) 
            res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                          approx_grad = True,
                                          bounds = bounds_M3, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
            
            print('VIEW_INDEPENDENT')
    
            #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
        
            #data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
            bounds_M4 = [(0,1),(.1,20),(0,2)]
        
            part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL) 
            res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                          approx_grad = True,
                                          bounds = bounds_M4, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
            
            
            print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
            
            #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        
        
            #data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
            bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL) 
            res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                          approx_grad = True,
                                          bounds = bounds_M5, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
            
    
            print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
            params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
            #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        
            #data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
            bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_ALL) 
            res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                          approx_grad = True,
                                          bounds = bounds_M6, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                          epsilon=epsilon_param)
            
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
            
            
            ##### RND Choice model
            rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
            
            re_evidence_subj = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]] + [rnd_choice]
            res_evidence[i] = re_evidence_subj
            
            ### Subject BF_LOG
            bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
            bf_log_group[i + '_BF_log'] = [bf_log_subj]
    
            
            ############################## Verbose == True ###########################
            
            verbose_debug = verbose_tot
            
            data_ALL_debug = data_ALL
            data_ALL_debug[-1] = True        
            
            if verbose_debug == True:
    
                # = data_M_debug[0]
                params_m_1 = res1[0]
                m_1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug, params_m_1)
                
                #data_M2_debug = data_M_debug[1]
                params_m_2 = res2[0]
                m_2 = VIEW_DEPENDENT(data_ALL_debug, params_m_2)
            
                #data_M3_debug = data_M_debug[2]
                params_m_3 = res3[0]
                m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_ALL_debug, params_m_3)
                
                #ädata_M4_debug = data_M_debug[3]
                params_m_4 = res4[0]
                m_4 = VIEW_INDEPENDENT(data_ALL_debug, params_m_4)
            
                #data_M5_debug = data_M_debug[4]
                params_m_5 = res5[0]
                m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_ALL_debug, params_m_5)
                
                #data_M6_debug = data_M_debug[5]
                params_m_6 = res6[0]
                m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_ALL_debug, params_m_6)
        
                res_debug = {models_names[0]: m_1,
                             models_names[1]: m_2,
                             models_names[2]: m_3,
                             models_names[3]: m_4,
                             models_names[4]: m_5,
                             models_names[5]: m_6}
                data_verbose_debug[i] = res_debug
                
        #### Get winning model trialwise dat ####
            #data_M1_debug = data_ALL_debug[0]
            params_m_1 = res1[0]
            res_M1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug, params_m_1)        
    
            trialwise_data[i] = res_M1[1]['data_store_1']
        restotal = res_evidence.sum(axis=1)
        cntrl_log = special.logsumexp(np.array(restotal[1::]))
        bf_log = (cntrl_log -(np.array(restotal[0])))
        
        results_1 = {'uncorr_LR_10':np.exp(-1*bf_log),
                    'subject_level_model_evidence':res_evidence,
                    'group_level_model_evidence':res_evidence.sum(axis=1),
                    'subject_level_uncorr_LR': bf_log_group,
                    'used_data': data_AB,
                    'subject_level_parameter-estimates':parameter_est,
                    'subject_level_trialwise_data_win_model':trialwise_data,
                    'baseline_model': np.sum(np.log([0.5 for i in range(189)]))}
        if verbose_tot==True:
            return (results_1,restotal,data_verbose_debug)
        elif verbose_tot==False:
            return results_1
        
        
    def behavioral_performance(self):
        
        import pandas as pd
        data_raw = self.clean_data 
        data_A = data_raw['A']
        data_B = data_raw['B']
        data_AB = {**data_A,**data_B}
        unq_ids = list(data_AB.keys())
        stimulus_len = len(self.groundtruth.index)
        data_perf_A = pd.DataFrame(index=['%_correct','n_errors','missings'])
        data_perf_B = pd.DataFrame(index=['%_correct','n_errors','missings'])

        m_performance = pd.DataFrame(index=['M_%_correct',
                                            'M_n_errors',
                                            'M_missings',])
        for vpn in unq_ids:
            curr_dat = data_AB[vpn]
            missings = stimulus_len-len(curr_dat.index)
            perc_correct = curr_dat['perf'].sum()/len(curr_dat.index)
            n_errors = len(curr_dat.index)-curr_dat['perf'].sum()
            if 'A' in vpn:
                data_perf_A[vpn] = [perc_correct,
                                  n_errors,
                                  missings]
            elif 'B' in vpn:
                data_perf_B[vpn] = [perc_correct,
                                    n_errors,
                                    missings]
        
        m_performance['M_A'] = list(data_perf_A.copy().mean(axis=1))
        m_performance['SD_A']= list(data_perf_A.copy().std(axis=1))
        m_performance['M_B']= list(data_perf_B.copy().mean(axis=1))
        m_performance['SD_B']= list(data_perf_B.copy().std(axis=1))
        
        results = {'res_A':data_perf_A,
                   'res_B':data_perf_B,
                   'M_res_AB':m_performance}
        self.behav_perf = results
        return results
            
    def learning_effect(self):
        
        data_raw = self.clean_data
        data_A = data_raw['A']
        data_B = data_raw['B']
        data_AB = {**data_A,**data_B}
        unq_ids = list(data_AB.keys())
        
        #for vpn in unq_ids
        
        
 
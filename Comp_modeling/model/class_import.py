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
        
 
    def fit_data_seperate(self, verbose_tot):
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,bic

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
        res_subj_BIC_proc = pd.DataFrame(index=models_names)
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
            parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
            
            ##### VIEW_DEPENDENT
            print('VIEW_DEPENDENT')
                    
            bounds_M2 = [(0,1),(.1,20),(0,2)]
            
            part_func_M2 = partial(VIEW_DEPENDENT,data_ALL,None) 
            res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                          approx_grad = True,
                                          bounds = bounds_M2, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                          epsilon=epsilon_param)
            
            parameter_est['VIEW_DEPENDENT'][i] = res2[0]
            
            ##### VIEW_DEPENDENTxCONTEXT_DEPENDENT
            print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

            bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
        
            part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_ALL,None) 
            res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                          approx_grad = True,
                                          bounds = bounds_M3, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
            
            ##### VIEW_INDEPENDENT
            print('VIEW_INDEPENDENT')
    
            bounds_M4 = [(0,1),(.1,20),(0,2)]
        
            part_func_M4 = partial(VIEW_INDEPENDENT,data_ALL,None) 
            res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                          approx_grad = True,
                                          bounds = bounds_M4, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
            
            ##### VIEW_INDEPENDENTxVIEW_DEPENDENT
            print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
                    
            bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_ALL,None) 
            res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                          approx_grad = True,
                                          bounds = bounds_M5, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
            
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
            
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
            
            
            ##### RND Choice model
            rnd_choice = np.sum([np.log(0.5) for i in range(len(VPN_output))])
            
            #### Collect model evidence
            res_ev_raw = [(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]]
            re_evidence_subj = res_ev_raw + [rnd_choice]
            res_evidence[i] = re_evidence_subj
            
            #### Calc subj BIC
            res_subj_BIC = []
            for model,ev in zip(models_names,res_ev_raw):
                n_params = len(list(parameter_est[model].keys()))
                sample_size = len(list(data_A.keys()))
                subj_BIC = bic(n_params, sample_size, ev)
                res_subj_BIC.append(subj_BIC)
            res_subj_BIC_proc[i] = res_subj_BIC + [rnd_choice]
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
                data_verbose_debug[i] = res_debug
                
        #### Get winning model trialwise dat ####
            params_m_1 = res1[0]
            res_M1 = VIEW_INDIPENDENTxCONTEXT(data_ALL_debug,None, params_m_1)        
    
            trialwise_data[i] = res_M1[1]['data_store_1']
        
        ### some calculation
        restotal = res_evidence.sum(axis=1)
        restotal_BIC = res_subj_BIC_proc.sum(axis=1)
        group_level_me_A = res_evidence[[i for i in res_evidence if 'A' in i]].sum(axis=1)
        group_level_me_B = res_evidence[[i for i in res_evidence if 'B' in i]].sum(axis=1)
        group_level_BIC_A = res_subj_BIC_proc[[i for i in res_subj_BIC_proc if 'A' in i]].sum(axis=1)
        group_level_BIC_B = res_subj_BIC_proc[[i for i in res_subj_BIC_proc if 'B' in i]].sum(axis=1)       
        
        ##### BIC group calc
        indxs = ['bic','raw_LL','n_params', 'sample_size']
        bic_fit_A= pd.DataFrame(index=indxs)
        bic_fit_B= pd.DataFrame(index=indxs)
        for time_data,time in zip([group_level_me_A,group_level_me_B],['A','B']):
            for model in models_names_bic:
                n_params = len(list(parameter_est[model].index))
                sample_size = len([i for i in res_evidence if 'A' in i])
                if time == 'A':
                    raw_LL = group_level_me_A[model]
                    bic_curr = bic(n_params, sample_size, raw_LL)
                    bic_fit_A[model] = [bic_curr,raw_LL,n_params, sample_size]
                elif time == 'B':
                    raw_LL = group_level_me_B[model]
                    bic_curr = bic(n_params, sample_size, raw_LL)
                    bic_fit_B[model] = [bic_curr,raw_LL,n_params, sample_size]  
        
        ### Group Post Model Prob
        post_model_win = pd.DataFrame()
        for time_data,time in zip([group_level_me_A,group_level_me_B],['A','B']):
            sum_subj_BIC = time_data.loc[models_names_bic[0]]
            curr_mod_prob = special.logsumexp(np.array(time_data))
            post_prob = np.exp(sum_subj_BIC-curr_mod_prob)
            post_model_win['win_model_' + time] = [post_prob]
        bf_log_group[i + '_BF_log'] = [bf_log_subj]
        ## summarize data
        results_1 = {'subject_level_model_evidence':res_evidence,
                     'res_subj_BIC_proc':res_subj_BIC_proc,
                     'group_level_me_A': group_level_me_A,
                     'group_level_me_B': group_level_me_B,
                     'group_level_me_AB':restotal,
                     'group_level_BIC_A':group_level_BIC_A,
                     'group_level_BIC_B':group_level_BIC_B,
                     'group_level_BIC_AB':restotal_BIC,
                     'used_data': data_AB,
                     'subject_level_parameter-estimates':parameter_est,
                     'subject_level_trialwise_data_win_model':trialwise_data,
                     'group_winmodel_post':post_model_win}
        self.fit_separate = results_1
        if verbose_tot==True:
            return (results_1,data_verbose_debug)
        elif verbose_tot==False:
            return results_1


    def fit_data_combined(self):
        
        import os
        os.chdir(self.path_to_modelfunctions)
        
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
        from tqdm import tqdm
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)
        ### Get data 
        data_unform = self.clean_data 
        data_A = data_unform['A']
        data_B = data_unform['B']
        #data_AB = {**data_A,**data_B}
        #### get unique IDS
        unique_id_t1 = list(data_A.keys())
        unique_id_t2 = list(data_B.keys())
        
        sample_answer_clms_t1 = [i+'_answer' for i in unique_id_t1]
        sample_answer_clms_t2 = [i+'_answer' for i in unique_id_t2]
        
        sample_perspective_clms = [i+'_perspective' for i in unique_id_t1]
        sample_perspective_clms = [i+'_perspective' for i in unique_id_t2]
        
        epsilon_param = 0.01
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
        
        for vpn_t1,vpn_t2 in zip(unique_id_t1,unique_id_t2):
            print(vpn_t1,vpn_t2)
            #func calls & rand starts
            
            curr_data_vpn_t1 = data_A[vpn_t1]
            curr_data_vpn_t2 = data_B[vpn_t2]
            # data import
            stim_ID_t1 = curr_data_vpn_t1['stim_IDs_VI'] #stimulus IDs of winning model
            stim_ID_t2 = curr_data_vpn_t2['stim_IDs_VI'] #stimulus IDs of winning model
            
            new_ID_t1 = curr_data_vpn_t1['new_IDs'] #trials where new ID is introduced 
            new_ID_t2 = curr_data_vpn_t2['new_IDs'] #trials where new ID is introduced 

            numb_prev_presentations_t1 = curr_data_vpn_t1['n_prev_VI'] #number_of_prev_presentations/view independent
            numb_prev_presentations_t2 = curr_data_vpn_t2['n_prev_VI'] #number_of_prev_presentations/view independent            
            
            stim_IDs_VD_t1 = curr_data_vpn_t1['stim_IDs_VD'] #view dependent IDs
            stim_IDs_VD_t2 = curr_data_vpn_t2['stim_IDs_VD'] #view dependent IDs
            
            stim_IDs_VI_t1 = curr_data_vpn_t1['stim_IDs_VI'] #view independent IDs
            stim_IDs_VI_t2 = curr_data_vpn_t2['stim_IDs_VI'] #view independent IDs
            
            VPN_output_t1 = curr_data_vpn_t1['answer'] #VPN answers
            VPN_output_t2  = curr_data_vpn_t2['answer'] #VPN answers
            verbose = False
            
            data_ALL_T1 = [VPN_output_t1,
                           new_ID_t1,
                           numb_prev_presentations_t1,
                           stim_IDs_VI_t1,
                           stim_IDs_VD_t1, 
                           verbose]
            data_ALL_T2 = [VPN_output_t2,
                           new_ID_t2, 
                           numb_prev_presentations_t2, 
                           stim_IDs_VI_t2, 
                           stim_IDs_VD_t2, 
                           verbose]

            #################### Model Optim      
            #i=vpn
    ####################  VIEW_INDIPENDENTxCONTEXT     ####################        
            def VIEW_INDIPENDENTxCONTEXT_t1_t2(data_M1_t1,data_M1_t2,params):
               res1_t1 = VIEW_INDIPENDENTxCONTEXT(data_M1_t1,None,params)
               res1_t2 = VIEW_INDIPENDENTxCONTEXT(data_M1_t2,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
           
            data_M1_t1 = data_ALL_T1
            data_M1_t2 = data_ALL_T2
            part_func_M1_t1_t2 = partial(VIEW_INDIPENDENTxCONTEXT_t1_t2,data_M1_t1,data_M1_t2) 
            bounds_M1= [(0,1),(0,1),(.1,20),(0,2)]
            res1_t1_t2 = optimize.fmin_l_bfgs_b(part_func_M1_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M1, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                          epsilon=epsilon_param)
    
            parameter_est['VIEW_INDIPENDENTxCONTEXT'][vpn_t1 + vpn_t2] = res1_t1_t2[0] 
    
        
    ####################  VIEW_DEPENDENT    ####################       
            def VIEW_DEPENDENT_t1_t2(data_M2_t1,data_M1_t2,params):
               res1_t1 = VIEW_DEPENDENT(data_M2_t1,None,params)
               res1_t2 = VIEW_DEPENDENT(data_M2_t1,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
            
            #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
            data_M2_t1 = data_ALL_T1
            data_M2_t2 = data_ALL_T2
    
        
            bounds_M2 = [(0,1),(.1,20),(0,2)]
            part_func_M2_t1_t2 = partial(VIEW_DEPENDENT_t1_t2,data_M2_t1,data_M2_t2)
            #part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
            res2_t1_t2  = optimize.fmin_l_bfgs_b(part_func_M2_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M2, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                          epsilon=epsilon_param)
            
            parameter_est['VIEW_DEPENDENT'][vpn_t1 + vpn_t2] = res2_t1_t2[0]
    
    # ####################  VIEW_DEPENDENTxCONTEXT_DEPENDENT    ####################       
    
            def VIEW_DEPENDENTxCONTEXT_DEPENDENT_t1_t2(data_M3_t1,data_M3_t2,params):
               res1_t1 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_t1,None,params)
               res1_t2 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_t2,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
            data_M3_t1 = data_ALL_T1
            data_M3_t2 = data_ALL_T2
     
            
            bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
        
            part_func_M3_t1_t2  = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT_t1_t2,
                                    data_M3_t1,
                                    data_M3_t2) 
            res3_t1_t2 = optimize.fmin_l_bfgs_b(part_func_M3_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M3, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                          epsilon=epsilon_param)
    
            parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][vpn_t1 + vpn_t2] = res3_t1_t2[0]
    
    
    # ####################  VIEW_INDEPENDENT    #################### 
    
            def VIEW_INDEPENDENT_t1_t2(data_M4_t1,data_M4_t2,params):
               res1_t1 = VIEW_INDEPENDENT(data_M4_t1,None,params)
               res1_t2 = VIEW_INDEPENDENT(data_M4_t2,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
    
            data_M4_t1 = data_ALL_T1
            data_M4_t2 = data_ALL_T2
    
            bounds_M4 = [(0,1),(.1,20),(0,2)]
        
            part_func_M4_t1_t2 = partial(VIEW_INDEPENDENT_t1_t2,data_M4_t1,data_M4_t2) 
            res4_t1_t2 = optimize.fmin_l_bfgs_b(part_func_M4_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M4, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENT'][vpn_t1 + vpn_t2] = res4_t1_t2[0]
    
     
    # ####################  VIEW_INDEPENDENTxVIEW_DEPENDENT    #################### 
    
            def VIEW_INDEPENDENTxVIEW_DEPENDENT_t1_t2(data_M5_t1,data_M5_t2,params):
               res1_t1 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_t1,None,params)
               res1_t2 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_t2,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
    
            data_M5_t1 = data_ALL_T1
            data_M5_t2 = data_ALL_T2
    
            bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M5_t1_t2 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT_t1_t2,data_M5_t1,data_M5_t2) 
            res5_t1_t2 = optimize.fmin_l_bfgs_b(part_func_M5_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M5, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                          epsilon=epsilon_param)
            parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][vpn_t1 + vpn_t2] = res5_t1_t2[0]
            
    
            
    # ####################  VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT    #################### 
    
            def VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_t1_t2(data_M6_t1,data_M6_t2,params):
               res1_t1 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_t1,None,params)
               res1_t2 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_t2,None,params)                     
               loss_t1 = res1_t1
               loss_t2 = res1_t2
               total_loss = loss_t1 + loss_t2
               return total_loss
    
            data_M6_t1 = data_ALL_T1
            data_M6_t2 = data_ALL_T2
    
            bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
        
            part_func_M6_t1_t2  = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_t1_t2,
                                          data_M6_t1,
                                          data_M6_t2) 
            res6_t1_t2  = optimize.fmin_l_bfgs_b(part_func_M6_t1_t2,
                                          approx_grad = True,
                                          bounds = bounds_M6, 
                                          x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                          epsilon=epsilon_param)
            
            #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
    
    # ####################  Random Choice #################### 
            def rnf_choice(VPN_output_t1,VPN_output_t2):
                n_trial_T1,n_trial_T2 = len(VPN_output_t1),len(VPN_output_t2)
                loss_T1 = np.sum([np.log(0.5) for i in range(n_trial_T1)])
                loss_T2 = np.sum([np.log(0.5) for i in range(n_trial_T2)])
                total_loss = loss_T1+loss_T2
                return total_loss
        
            rnd_choice_t1_t2 = rnf_choice(VPN_output_t1,VPN_output_t2)
            res_total = [res1_t1_t2,
                         res2_t1_t2,
                         res3_t1_t2,
                         res4_t1_t2,
                         res5_t1_t2,
                         res6_t1_t2]
            re_evidence_subj = np.array([(-1*i[1]) for i in res_total]+[rnd_choice_t1_t2])
    
            res_evidence[vpn_t1 + vpn_t2] = re_evidence_subj   
               
        
        
        self.combined_fit = res_evidence
        return res_evidence


    def RFX_modelselection(self):       
        import matlab.engine
        import numpy as np
        eng = matlab.engine.start_matlab()
        #see https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/#between-conditions-rfx-bms
        data_raw = self.fit_separate['res_subj_BIC_proc'].copy()
        data_clms = [i for i in data_raw.columns]
        a_t = data_raw[[i for i in data_clms if 'A' in i]]
        b_t = data_raw[[i for i in data_clms if 'B' in i]]
        
        # a_t = fit_data_sample_T1['subject_level_model_evidence'].copy()
        # b_t = fit_data_sample_T2['subject_level_model_evidence'].copy()
        # mt array(rows,columns,sheets)
        # RFX-array(models,subjects,cond)
    
        data_rfx_cond= np.zeros([7,3,2])
        data_rfx_cond[:,:,0] = [[j for j in i[1]] for i in a_t.iterrows()]
        data_rfx_cond[:,:,1] = [[j for j in i[1]] for i in b_t.iterrows()]
        data_rfx_cond_m = matlab.double(data_rfx_cond.tolist())
        
        # non-exceedance probability of difference in models across conditions
        non_exceedence_prob = eng.VBA_groupBMC_btwConds(data_rfx_cond_m)
        # exceedance probability of no larger difference in models across conditions
        #ex_prob = 1-float(non_exceedence_prob)
        return ('non_exceedence_prob',non_exceedence_prob)

        
    def behavioral_performance(self):
        import pandas as pd
        import numpy as np
        
        
        data_raw = self.clean_data 
        time_points = ['A','B']
        data_A = data_raw[time_points[0]]
        data_B = data_raw[time_points[1]]
        data_AB = {**data_A,**data_B}
        unq_ids = list(data_AB.keys())
        stimulus_len = len(self.groundtruth.index)
        data_perf_A = pd.DataFrame(index=['%_correct','n_errors','missings'])
        data_perf_B = pd.DataFrame(index=['%_correct','n_errors','missings'])

        m_performance = pd.DataFrame(index=['M_%_correct',
                                            'M_n_errors',
                                            'M_missings',])
        quality_crit = pd.DataFrame()
        for vpn in unq_ids:
            #get dataframe per subj
            curr_dat = data_AB[vpn]
            
            ## egt quality indices
            missings = stimulus_len-len(curr_dat.index)
            perc_correct = (curr_dat['perf'].sum()/len(curr_dat.index))*100
            n_errors = len(curr_dat.index)-curr_dat['perf'].sum()
            
            ## get performance criterion
            q_crit_10_11_12 = curr_dat.loc[curr_dat['n_prev_VI'] >= 10]
            q_crit_10_11_12_M = np.around(q_crit_10_11_12['perf'].mean(),decimals=2)
            
            if 'A' in vpn:
                data_perf_A[vpn] = [perc_correct,
                                    n_errors,
                                    missings]
                quality_crit[vpn] = [q_crit_10_11_12_M]
            elif 'B' in vpn:
                data_perf_B[vpn] = [perc_correct,
                                    n_errors,
                                    missings]
                quality_crit[vpn] = [q_crit_10_11_12_M]
        
        m_performance['M_A'] = list(data_perf_A.copy().mean(axis=1))
        m_performance['SD_A']= list(data_perf_A.copy().std(axis=1))
        m_performance['M_B']= list(data_perf_B.copy().mean(axis=1))
        m_performance['SD_B']= list(data_perf_B.copy().std(axis=1))
        
        results = {'res_A':data_perf_A,
                   'res_B':data_perf_B,
                   'M_res_AB':m_performance}
        self.behav_perf = results
        return {'results':results,
                'used_data':data_raw,
                'q_crit': quality_crit}
    
            
    def learning_effect(self):       
        
        import pingouin as pg
        import pandas as pd
        data_raw = self.clean_data
        data_A = data_raw['A']
        data_B = data_raw['B']
        data_AB = {**data_A,**data_B}
        unq_ids = list(data_AB.keys())
        res_total = {}
        #### learning effect for correct answers and for total 'yes' answers
        for variable in ['perf', 'answer']:
            df_1_3_L_A = []
            df_10_10_L_A = []
            df_1_3_L_B = []
            df_10_10_L_B = []
            #['perf', 'answer']
            #variable = 'answer'
            for vpn in unq_ids:
                curr_dat = data_AB[vpn]
                df_1_2_3 = curr_dat.loc[curr_dat['n_prev_VI'] <=3][variable].sum()/len(curr_dat.loc[curr_dat['n_prev_VI'] <=3][variable].index)
                df_10_11_12 = curr_dat.loc[curr_dat['n_prev_VI'] >=10][variable].sum()/len(curr_dat.loc[curr_dat['n_prev_VI'] >=10][variable].index)
                if 'A' in vpn:   
                    df_1_3_L_A.append(df_1_2_3)
                    df_10_10_L_A.append(df_10_11_12)
                elif 'B' in vpn:
                    df_1_3_L_B.append(df_1_2_3)
                    df_10_10_L_B.append(df_10_11_12)
                    
            p_answer = pd.DataFrame(index=['1','2','3'])
            p_answer['A_1_3'] = df_1_3_L_A
            p_answer['B_1_3'] = df_1_3_L_B
            p_answer['A_10_12'] = df_10_10_L_A
            p_answer['B_10_12'] = df_10_10_L_B
            
            
            res_A = pg.ttest(p_answer['A_10_12'],p_answer['A_1_3'],paired = True)
            res_B = pg.ttest(p_answer['B_10_12'],p_answer['B_1_3'],paired = True)
            
            res = pd.DataFrame(index = [i for i in res_A.columns])
            res['A_1_2_3_vs_10_11_12'] = res_A.loc['T-test']
            res['B_1_2_3_vs_10_11_12'] = res_B.loc['T-test']
            

            mean_pyes = pd.DataFrame(index = ['%_M','%_SD'])
            for i in p_answer:
                mean_pyes[i] = [p_answer[i].mean(),p_answer[i].std()]
                                
            tot_res = {'used_dat':data_raw,
                       'proc_data':p_answer,
                       'res_ttest':res,
                       'descriptives':mean_pyes}
            res_total[variable] = tot_res
            
        ##### AT_2013 effectsize
        # reconstruct effectsize from apps&tsakiris study
        observed_D = pg.compute_effsize_from_t(tval = 14.661,
                                               N=16,
                                               eftype='cohen')
        observed_power = pg.power_ttest(d=observed_D,
                                        power=None,
                                        alpha =.0001,
                                        n=16)   
         
        req_N = pg.power_ttest(d=observed_D,
                               power=.90,
                               alpha =.02,
                               n=None)   
         
        power_analysis = {'AT_observed_tval':14.661,
                          'AT_N':16,
                          'AT_recalc_effectsize':observed_D,
                          'req_N_for_replication':req_N,
                          'observed_power': observed_power}
        res_total['AT_ES_D_poweranalysis'] = power_analysis
        
        
        ##### Our analysis -> Only on CORRECT Yes-Answers
        # 
        res_tt_correct = pd.DataFrame(res_total['perf']['res_ttest']['A_1_2_3_vs_10_11_12'])
        res_tt_correct['B_1_2_3_vs_10_11_12'] = res_total['perf']['res_ttest']['B_1_2_3_vs_10_11_12']
        es_D_T1 = res_total['perf']['res_ttest']['A_1_2_3_vs_10_11_12'].loc['cohen-d'] 
        es_D_T2 = res_total['perf']['res_ttest']['B_1_2_3_vs_10_11_12'].loc['cohen-d'] 
        # arch_pwr_T1 = res_total['perf']['res_ttest']['A_1_2_3_vs_10_11_12'].loc['power'] 
        # arch_pwr_T2 = res_total['perf']['res_ttest']['B_1_2_3_vs_10_11_12'].loc['power']
        # pval_T1 = res_total['perf']['res_ttest']['A_1_2_3_vs_10_11_12'].loc['p-val'] 
        # pval_T2 = res_total['perf']['res_ttest']['B_1_2_3_vs_10_11_12'].loc['p-val']
        # pval_T1 = res_total['perf']['res_ttest']['A_1_2_3_vs_10_11_12'].loc['T'] 
        # pval_T2 = res_total['perf']['res_ttest']['B_1_2_3_vs_10_11_12'].loc['T'] 
        
        req_N_T1 = pg.power_ttest(d=es_D_T1,
                                  power=.90,
                                  alpha =.02,
                                  n=None)   
        req_N_T2 = pg.power_ttest(d=es_D_T2,
                                  power=.90,
                                  alpha =.02,
                                  n=None) 
          
        power_analysis_perf = {'req_N_T1_perf':req_N_T1,
                               'res_t':res_tt_correct,
                               'req_N_T2_perf':req_N_T2}
        res_total['req_n_performance'] = power_analysis_perf
        return res_total

    
    def task_reliability(self):
        # ICC
        res_beh = self.behav_perf
        icc_data = res_beh['res_A'].T.append(res_beh['res_B'].T).round(3)
        
        icc_data['time'] = [i[-1] for i in icc_data.index]
        icc_data['id'] = [i[0] for i in icc_data.index]
        
        import pingouin as pg
        icc = pg.intraclass_corr(data=icc_data, 
                                 targets='id', 
                                 raters='time',
                                 ratings='%_correct')
        icc_2 = icc.loc[icc['Type'] == 'ICC2']
        #pearson_corr
        corr_data = icc_data
        corr_data_A = corr_data.loc[corr_data['time'] == 'A']['%_correct']
        corr_data_B = corr_data.loc[corr_data['time'] == 'B']['%_correct']
        corr = pg.corr(corr_data_A, corr_data_B, tail='two-sided', method='pearson')
        
        res = {'icc': icc_2,
              'corr': corr}
        return res
    
    
    def model_selection_AT(self):
        import pandas as pd
        import numpy as np
        from scipy import special
        ########## Data Transcribed from paper
        
        ##### Model_selection
        # Formulas from: 
        #   Glover, S., Dixon, P. 
        #   Likelihood ratios: A simple and flexible statistic for empirical psychologists. 
        #   Psychonomic Bulletin & Review 11, 791–806 (2004). 
        #   https://doi.org/10.3758/BF03196706
        
        ll_raw = {'View-dependent': [-1590,3],
                  'View-dependent_context-dependent': [-1510,4],
                  'View-INdependent': [-1480,3],
                  'View-independent_context-dependent': [-1425,4],
                  'View-independent_View-dependent': [-1475,5],
                  'View-independent_View-dependent_context-dependent': [-1430,6],
                  }
        ll_dat = pd.DataFrame.from_dict(data=ll_raw)
        ll_dat['indx']=['ll','n_param']
        ll_dat.set_index(keys='indx',inplace=True)
        
        model_win = 'View-independent_context-dependent'
        model_cntrl = [i for i in ll_dat if i != model_win]
        model_all = [i for i in ll_dat]
        
        ll_fin = pd.DataFrame(index = ['LR','LR_corr'])
        n_size = 16
        
        
        for i in model_cntrl:
            # Get Log-Likelihoods
            cntrl_ll = ll_dat[i][0]
            win_ll = ll_dat[model_win][0]
            
            # Get n_params
            win_nparam = ll_dat[model_win][1]
            cntrl_nparam = ll_dat[i][1]
            # uncorr LR
            lr_var = -2*(cntrl_ll-win_ll)
            # corr LL -> Glover & Dixon, 2004
            lr_corr = lr_var*((np.exp((cntrl_nparam-win_nparam)))**(np.log(n_size)/2))
            
            # POSITIVE VALUES FAVOR WINNING MODEL
            ll_fin[i] = [lr_var]+[lr_corr]
        
        # Compare Multiple Models, lambda_mult     
        lmbda_mult = -2*(special.logsumexp(np.array(ll_dat.T['ll'])-ll_dat[model_win][0]))
        
        #transpose 
        #get BIC
        bic = []
        for i in model_all:
            raw_ll = ll_dat[i][0]
            nparam = ll_dat[i][1]
            bic_raw = (nparam*np.log(n_size))-(2*raw_ll)
            bic.append(bic_raw)
        
        ll_dat = ll_dat.T
        ll_dat['BIC'] = bic
        return {'AT_model_selection_results_nparams': ll_dat,
                'lambda_mult_model_selection':lmbda_mult,
                'LR_fin': ll_fin}  


    def corr_LR(self):
        import pandas as pd
        import numpy as np
        
        fit_data_seperate = self.fit_separate
        #get names
        name_models_LR = [i for i in fit_data_seperate['subject_level_parameter-estimates']]
        #get number of params
        n_params = pd.DataFrame()
        for i in name_models_LR:
            n_par = fit_data_seperate['subject_level_parameter-estimates'][i].shape[0]
            n_params[i] = [n_par]
        #get number of subjects
        n_subj = len([i for i in fit_data_seperate['subject_level_model_evidence']])
            
        res_lr_A = pd.DataFrame(index = ['lr','corr_lr','correction'])
        res_lr_B = pd.DataFrame(index = ['lr','corr_lr','correction'])
        group_ev = []
        for time in ['A','B']:
            curr_dat_raw = fit_data_seperate['subject_level_model_evidence']
            curr_clmns = [i for i in list(curr_dat_raw.columns) if time in i]
            curr_dat = curr_dat_raw[curr_clmns]
            group_lvl_ev = curr_dat.sum(axis=1)
            group_ev.append(group_lvl_ev)
            for model_LR in name_models_LR[1::]:
                #winning model
                model_ev_win = np.array(group_lvl_ev[name_models_LR[0]])
                model_ev_win_Nparam = np.array(n_params[name_models_LR[0]])
                #control models
                model_ev_cntrl = np.array(group_lvl_ev[model_LR])
                model_ev_cntrl_Nparam = np.array(n_params[model_LR])
                #uncorrected LR10
                lr_uncorr = model_ev_win-model_ev_cntrl
                #corrected LR10
                exp_modelparam = np.exp(model_ev_win_Nparam-model_ev_cntrl_Nparam)
                correction = np.log(exp_modelparam**(np.log(n_subj)/2))
                corr_lr = lr_uncorr + correction
                res_list = [float(lr_uncorr),float(corr_lr),float(correction)]
                if time == 'A':
                    res_lr_A[str(name_models_LR[0]) + '_vs_' + str(model_LR)] = res_list
                elif time == 'B':
                    res_lr_B[str(name_models_LR[0]) + '_vs_' + str(model_LR)] = res_list
            used_data_lr = {'n_subj':n_subj,
                         'n_params':n_params,
                         'fit_data':fit_data_seperate}
        return {'corr_lr_A':res_lr_A ,
                'corr_lr_B':res_lr_B ,
                'used_data':used_data_lr}
            
    

    def time_agn10_LR(self):
        ''' time-agnostic log-likelihood difference converted to normal 
        probability i.e. relative likelihood. Shows the difference in probability between observing the 
        data under a common set of params vs. under seperate sets of params.
        A relative likelihood of >1 indicates a better model-fit for the 
        combined ML params'''
        import numpy as np
        import pandas as pd
        ### time agn LR with BIC correction
        res_fit_sep_curr = self.fit_separate['subject_level_model_evidence'].sum(axis=1)
        res_fit_comb = self.combined_fit.sum(axis=1)
        model_names = list(res_fit_sep_curr.index)
        res_time_agn10 = pd.DataFrame(index = ['fit_comb','fit_sep','LR_agn_10'])
        for model in model_names:
            curr_sep = res_fit_sep_curr.loc[model]
            curr_comb = res_fit_comb.loc[model]
            curr_agn_10 = np.around(np.exp(curr_comb-curr_sep),decimals = 2)
            res_time_agn10[model] = np.array([curr_comb, curr_sep, curr_agn_10])
        return {'results': res_time_agn10,
                'res_fit_sep_curr':res_fit_sep_curr,
                'res_fit_comb':res_fit_comb
                
                }
            
    def fit_data_separate_LOOCV(self):
        ##########
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        from joblib import Parallel, delayed
        import os
        import pandas as pd
        np.random.seed(1993)
        from model_functions import data_cv
        ### Get data 
        data_unform = self.clean_data
        data_A = data_unform['A']
        data_B = data_unform['B']
        data_AB = {**data_A,**data_B}
        ### data reformating into np.arrays
        ids_t1_t2_joblib = [i for i in data_AB.keys()]
        data_job = []
        
        for ids in ids_t1_t2_joblib:
            data = data_AB[ids]
            data_np = data_AB[ids].to_numpy()
            data_np[:,1] = data_np[:,1].astype(int)
            data_descr = [i for i in data_AB[ids]]
            #(unique_id, data, lbfgs_epsilon, verbose_tot)
            data_job.append([ids, data_np, 0.01, False])
        
        #### get unique IDS
        unique_id = list(data_AB.keys())
        os.chdir(self.path_to_modelfunctions)

        if __name__ == 'class_import':   
            results123 = Parallel(n_jobs=-1,
                                  verbose=50,
                                  backend='loky')(delayed(data_cv)(i) for i in data_job)    
            
            index_models = [i for i in results123[0][1]]
            data_res_tot = pd.DataFrame(index = index_models)
            for vpn_dat in results123:
                curr_id,curr_dat = vpn_dat[0],vpn_dat[1]
                model_ev_L = []
                for model in index_models:
                    model_ev_L.append(curr_dat[model])
            
                data_res_tot[curr_id] = model_ev_L
            group_level_CV_A = data_res_tot[[i for i in data_res_tot if 'A' in i]].sum(axis=1)
            group_level_CV_B = data_res_tot[[i for i in data_res_tot if 'B' in i]].sum(axis=1)
            data_res_tot['A_sum'] = data_res_tot[[i for i in data_res_tot if 'A' in i]].sum(axis=1)
            data_res_tot['B_sum'] = data_res_tot[[i for i in data_res_tot if 'B' in i]].sum(axis=1)
            return_dat = {'raw_results_CV':     results123,
                          'subj_level_CV':      data_res_tot,
                          'group_level_CV_A':   group_level_CV_A,
                          'group_level_CV_B':   group_level_CV_B}
                          
            return return_dat     
        
    def ttest_procedure(self):
        import pandas as pd
        import pingouin as pg
        import numpy as np
        ''' 
        As described in Apps & Tsakiris, we ran a series of independent
        two-sided t-tests and corrected via Benjamini–Hochberg false discovery
        rate (FDR) on the BIC
        '''
        
        data_fit = self.fit_separate['res_subj_BIC_proc']
        data_A = data_fit[[i for i in data_fit if 'A' in i]]
        data_B = data_fit[[i for i in data_fit if 'B' in i]]
        
        win_model = 'VIEW_INDIPENDENTxCONTEXT'
        indx_models = [i for i in data_B.T if 'VIEW_INDIPENDENTxCONTEXT' not in i]
        indx_res = ['p-val','cohen-d','BF10','obs_power','T']
        
        res_A = pd.DataFrame(index = indx_res)
        res_B = pd.DataFrame(index = indx_res)
        #for each time-point
        for time,time_str in zip([data_A, data_B],['data_A', 'data_B']):
            data_curr = time.copy().T
            for model in indx_models:
                res_tt_raw = pg.ttest(data_curr[win_model],data_curr[model])
                res_tt_ind = [res_tt_raw['p-val'][0],
                              res_tt_raw['cohen-d'][0],
                              res_tt_raw['BF10'][0],
                              res_tt_raw['power'][0],
                              res_tt_raw['T'][0]]
                
                clm_name = win_model + '_vs_' + model
                if time_str == 'data_A':
                    res_A[clm_name] = res_tt_ind
                elif time_str == 'data_B':
                    res_B[clm_name] = res_tt_ind
        #FDR correction
        fdr_dat_A = res_A.copy().T.astype(float)
        fdr_dat_A['pval_FDR'] = pg.multicomp([i for i in fdr_dat_A['p-val']],
                                             alpha=0.02,
                                             method='fdr_bh')[1]
        fdr_dat_B = res_B.copy().T.astype(float)
        fdr_dat_B['pval_FDR'] = pg.multicomp([i for i in fdr_dat_B['p-val']],
                                             alpha=0.02,
                                             method='fdr_bh')[1]
        
        return_dict={'t_test_A':fdr_dat_A.T,
                     't_test_B':fdr_dat_B.T}
        return return_dict
        
    ###                     (int,                   int,          tuple(min,max))
    def comp_model_pwr(self, synth_data_per_subject, n_subsamples,range_samplesize,save_dat_path):
        from tqdm import tqdm
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)
        import glob
        import random 
        from joblib import Parallel, delayed
        import os
        ##### location of data
        path_data = self.path_to_data
        ##### loc of class
        path_to_class = self.path_to_modelfunctions
        ### loc of ground truth
        path_ground_truth = self.path_to_groundtruth
        os.chdir(path_to_class)
        
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
        
        
        def ttest_procedure_pwr(subject_level_model_evidence):
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
        
    
    
        ### init class with paths
        res_abba = power_analyses(path_data,path_ground_truth,path_to_class)
        
        ### get pilot_data
        res_beh_power1 = res_abba.get_data(True)
        
        
        
        ########## DOMINIC IDEA 
        '''use View-IndependentxContext model fitted with pilot data to generate
        artificial data'''
        
        ### generate N datasets from each fitted model
        synth_dat_per_subj = synth_data_per_subject
        synth_data_gen = res_abba.gen_data(synth_dat_per_subj)
        
        ### set params
        n_subsamples = n_subsamples # N of subsamples
        ss_min,ss_max = range_samplesize[0],range_samplesize[1]
        samplesizes_it_raw = [i for i in range(ss_min,ss_max)] # size of samples
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
                        res_gen_data1234_TT = ttest_procedure_pwr(tt_data)
                        
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
        
        
        
        
        
        ##### Save Data #####
        params_summ = pd.DataFrame()
        params_summ['n_subsamples'] = [n_subsamples]
        params_summ['synth_dat_per_subj'] = [synth_dat_per_subj]
        
        #save_dat_path = r'C:\Users\de_hauk\HESSENBOX\FRAPPS_RR (Nestor Israel Zaragoza Jimenez)\Power_analysis\res'
        os.chdir(save_dat_path)
        
        dataA1 = res_total_power['gen_view_dep']['A']['summ']
        dataA2 = res_total_power['gen_view_ind_context']['A']['summ']
        dataB1 = res_total_power['gen_view_dep']['B']['summ']
        dataB2 = res_total_power['gen_view_ind_context']['B']['summ']
        
        dataA1.to_csv(save_dat_path+'/gen_view_dep_A_summ.csv', 
                      sep=',',
                      index = True)
        dataA2.to_csv(save_dat_path+'/gen_view_ind_context_A_summ.csv', 
                      sep=',',
                      index = True)
        dataB1.to_csv(save_dat_path+'/gen_view_dep_B_summ.csv', 
                      sep=',',
                      index = True)
        dataB2.to_csv(save_dat_path+'/gen_view_ind_context_B_summ.csv', 
                      sep=',',
                      index = True)
        
        params_summ.to_csv(save_dat_path+'/params_summ.csv', 
                      sep=',',
                      index = True)
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")
        fig, axs = plt.subplots(2,2,sharey = True,sharex = True)
        ### fontsizes
        fs_xticks = 16
        fs_yticks = fs_xticks
        fs_axs_titles = 20
        
        
        
        fig.suptitle('Posterior model probability of the winning model given generative mechanisms', fontsize=fs_axs_titles+2)
        fig.set_size_inches(20, 12, forward=True)
        plte = 'Blues'
        
        
        
        ### gen view-ind X context
        sns.boxplot(data=dataA2, palette = plte, ax = axs[0,0])
        
        axs[0,0].set_title('T 1', fontsize=fs_axs_titles)
        axs[0,0].set_ylabel('view-independent X context', fontsize=fs_axs_titles)
        
        
        sns.boxplot(data=dataB2, palette = plte,ax = axs[0,1] )
        axs[0,1].set_title('T 2', fontsize=fs_axs_titles)
        
        ### gen view-dependent
        #A
        sns.boxplot(data=dataA1, palette = plte,ax = axs[1,0])
        
        #B
        sns.boxplot(data=dataB1, palette = plte,ax = axs[1,1] )
        axs[1,0].set_ylabel('view-dependent', fontsize=fs_axs_titles)
        axs[1,0].set_xlabel('sample size', fontsize=fs_axs_titles)
        axs[1,1].set_xlabel('sample size', fontsize=fs_axs_titles)
        
        
        axs[1,1].tick_params(axis='both', which='major', labelsize=fs_xticks)
        axs[1,0].tick_params(axis='both', which='major', labelsize=fs_xticks)
        axs[0,0].tick_params(axis='both', which='major', labelsize=fs_xticks)
        
        fig.savefig('unlimited_power.png', dpi=300)
        return res_total_power
    
    
    
    






        
       

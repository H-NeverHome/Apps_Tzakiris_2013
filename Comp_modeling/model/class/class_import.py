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
                     'B':final_dat_B}
        self.clean_data = final_dat
        if verbose == True:
            return final_dat

    def fit_data(self):
        import os
        os.chdir(self.path_to_modelfunctions)
        from model_functions import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
        from model_functions import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
        from model_functions import VIEW_INDIPENDENTxCONTEXT_CV,VIEW_DEPENDENT_CV
        from model_functions import VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV,VIEW_INDEPENDENT_CV
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENT_CV
        from model_functions import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV
    
        
        
        from tqdm import tqdm
        import pandas as pd
        import numpy as np
        from functools import partial
        from scipy import optimize,special
        np.random.seed(1993)
        ### Get data 
        data = self.clean_data
        
        #### get unique IDS
        unique_id = list(data.keys())
        sample_answer_clms = [i+'_answer' for i in unique_id]
        sample_perspective_clms = [i+'_perspective' for i in unique_id]
        
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
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
        
        parameter_est = {'VIEW_INDIPENDENTxCONTEXT': pd.DataFrame(index=params_M1_name),
                        'VIEW_DEPENDENT': pd.DataFrame(index=params_M2_name),
                        'VIEW_DEPENDENTxCONTEXT_DEPENDENT': pd.DataFrame(index=params_M3_name),
                        'VIEW_INDEPENDENT': pd.DataFrame(index=params_M4_name),
                        'VIEW_INDEPENDENTxVIEW_DEPENDENT': pd.DataFrame(index=params_M5_name),
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT': pd.DataFrame(index=params_M6_name)
                            }
    
        total_results = {}
        
        
        for vpn in unique_id:
            print(vpn)
            curr_data_vpn = data[vpn]
            
            cv_score_view_ind_cont = []
            cv_score_view_dep = []
            cv_score_view_dep_cont=[]
            cv_score_view_ind = []
            cv_score_view_ind_dep = []
            cv_score_view_ind_dep_cont = []
            
            # for trial in trial
            
            trials_n = int(curr_data_vpn.shape[0]-1)
            
            #### debug
            test_L = []
            hold_L = []
            
            
            
            
            for indx in tqdm(range(trials_n)):
                
                # print(vpn)

                # curr_data_vpn = data[vpn]
    
                # # get data
                # stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
                # new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
                # n_prev_pres=    np.array(curr_data_vpn['n_prev_VI'])    #number_of_prev_presentations
                # stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
                # VPN_output =    np.array(curr_data_vpn['answer'])       #VPN answers
                # verbose =       False
    
                # data_ALL = [VPN_output.astype(int), 
                #             new_ID.astype(int), 
                #             n_prev_pres.astype(int), 
                #             stim_IDs_VI, 
                #             stim_IDs_VD, 
                #             verbose]
                # holdout-data
                holdout_data = curr_data_vpn.copy().loc[indx]
                action = holdout_data[vpn+'_answer'].copy()
                # testing data
                test_data = curr_data_vpn.copy().drop(axis=0,index = indx).reset_index()
                curr_index = indx-1
                # data import
                stim_IDs = test_data['stim_IDs'].copy()    #stimulus IDs of winning model 
                new_ID = test_data['new_IDs'].copy()      #trials where new ID is introduced 
                numb_prev_presentations = test_data['n_prev_pres'].copy()      #number_of_prev_presentations
                stim_IDs_perspective = test_data[vpn+'_perspective'].copy()    #view dependent
                VPN_output = test_data[vpn+'_answer'].copy()       #VPN answers
                verbose = False
                
                ##### Model Optim
                test_L.append(holdout_data)
                hold_L.append(test_data)
    ########################################## VIEW_INDIPENDENTxCONTEXT #####################################             
    
                data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
        
                bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
                
                part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
                res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                              approx_grad = True,
                                              bounds = bounds_M1, 
                                              x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                              epsilon=epsilon_param)
    
                data_M1[-1] = True 
                data_M1_debug = data_M1.copy()
                params_m_1 = res1[0]
                m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
                
                init_V_m_1 = 0
                init_C_m_1 = 0
                if indx == 0:
                    init_V_m_1 += m_1[1]['init_val']['init_v']
                else:
                    data_cv_score = m_1[1]['data_store_1'].loc[curr_index]
                    init_V_m_1 += data_cv_score['history_V']
                    init_C_m_1 += data_cv_score['history_C']
        
    
                cv_trial_indeXcontext = VIEW_INDIPENDENTxCONTEXT_CV(params_m_1,
                                                                    init_V_m_1,
                                                                    init_C_m_1,
                                                                    action)
                
                
    ###################################### VIEW_DEPENDENT ######################################            
    
    
                data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
                
                data_M2_debug = data_M2.copy()
                data_M2_debug[-1] = True 
                bounds_M2 = [(0,1),(.1,20),(0,2)]
                
                part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
                res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                              approx_grad = True,
                                              bounds = bounds_M2, 
                                              x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                              epsilon=epsilon_param)
                params_m_2 = res2[0]
                #parameter_est['VIEW_DEPENDENT'][vpn] = res2[0]
        
                
                m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
    
                init_V_m_2 = 0
                if indx == 0:
                    init_V_m_2 += m_2[1]['init_val']
    
                else:
                    data_cv_score = m_2[1]['history_V_cv'][curr_index]
                    init_V_m_2 += data_cv_score
    
                cv_trial_dep = VIEW_DEPENDENT_CV(params_m_2,
                                                 init_V_m_2,
                                                 action)
    
    ######################################### VIEW_DEPENDENTxCONTEXT_DEPENDENT ##################################### 
                
                data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
                bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
            
                part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
                res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                              approx_grad = True,
                                              bounds = bounds_M3, 
                                              x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                              epsilon=epsilon_param)
                #parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][vpn] = res3[0]
    
    
                data_M3_debug = data_M3.copy()
                data_M3_debug[-1] = True 
                params_m_3 = res3[0]
                m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
                
                init_C_m_3 = 0
                init_V_m_3 = 0
                if indx == 0:
                    init_V_m_3 += m_3[1]['suppl']
    
                else:
                    init_C_m_3 = m_3[1]['history_V_dep'][curr_index]
                    init_V_m_3 = m_3[1]['history_C'][curr_index]
    
    
                cv_trial_dep_cont = VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV(params_m_3,
                                                                   init_V_m_3,
                                                                   init_C_m_3,
                                                                   action)
    ######################################## VIEW_INDEPENDENT ######################################
    
                data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
                bounds_M4 = [(0,1),(.1,20),(0,2)]
            
                part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
                res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                              approx_grad = True,
                                              bounds = bounds_M4, 
                                              x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                              epsilon=epsilon_param)
                #parameter_est['VIEW_INDEPENDENT'][vpn] = res4[0]
                
                
                                
                data_M4_debug = data_M4.copy()
                data_M4_debug[-1] = True
                params_m_4 = res4[0]
                m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
    
                init_V_m_4 = 0
                if indx == 0:
                    init_V_m_4 += m_4[1]['suppl']
    
                else:
                    init_V_m_4 = m_4[1]['history_total'][curr_index]
                
                cv_trial_ind = VIEW_INDEPENDENT_CV(params_m_4, init_V_m_4, action)
      
                
    ################################## VIEW_INDEPENDENTxVIEW_DEPENDENT ############################################
    
                data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
                bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
            
                part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
                res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                              approx_grad = True,
                                              bounds = bounds_M5, 
                                              x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                              epsilon=epsilon_param)
                #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][vpn] = res5[0]
                    
                data_M5_debug = data_M5.copy()
                data_M5_debug[-1] = True
                params_m_5 = res5[0]
                m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
    
                init_V_ind = 0
                init_V_dep = 0
                if indx == 0:
                    init_V_ind += m_5[1]['suppl'][0]
                    init_V_dep += m_5[1]['suppl'][1]
                else:
                    init_V_ind += m_5[1]['history_V_depend_L'][curr_index]
                    init_V_dep += m_5[1]['history_V_independ_L'][curr_index]
                
                cv_trial_ind_dep = VIEW_INDEPENDENTxVIEW_DEPENDENT_CV(params_m_5, init_V_dep,init_V_ind, action)
      
    ########################### VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT ###################################################     
    
                data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
                bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
            
                part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
                res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                               approx_grad = True,
                                               bounds = bounds_M6, 
                                               x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                               epsilon=epsilon_param)
                
                    
                data_M6_debug = data_M6.copy()
                data_M6_debug[-1] = True
                params_m_6 = res6[0]
                m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)
    
                init_V_ind = 0
                init_V_dep = 0
                init_C = 0
                if indx == 0:
                    init_V_ind += m_6[1]['suppl'][0]
                    init_V_dep += m_6[1]['suppl'][1]
                else:
                    init_C += m_6[1]['history_C'][curr_index]
                    init_V_ind += m_6[1]['history_V_depend_L'][curr_index]
                    init_V_dep += m_6[1]['history_V_independ_L'][curr_index]
                                                                                    
                cv_trial_ind_dep_cont = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV(params_m_6,
                                                                                   init_V_dep,
                                                                                   init_V_ind,
                                                                                   init_C, 
                                                                                   action)
                
                #parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][vpn] = res6[0]
    
    ##############################################################################        
                
                cv_score_view_ind_cont.append(cv_trial_indeXcontext)
                cv_score_view_dep.append(cv_trial_dep)
                cv_score_view_dep_cont.append(cv_trial_dep_cont)
                cv_score_view_ind.append(cv_trial_ind)
                cv_score_view_ind_dep.append(cv_trial_ind_dep)
                cv_score_view_ind_dep_cont.append(cv_trial_ind_dep_cont)
    
            df_index = ['VIEW_INDIPENDENTxCONTEXT',
                        'VIEW_DEPENDENT',
                        'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                        'VIEW_INDEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                        'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT']
            
            df_data = [np.sum(cv_score_view_ind_cont),
                       np.sum(cv_score_view_dep),
                       np.sum(cv_score_view_dep_cont),
                       np.sum(cv_score_view_ind),
                       np.sum(cv_score_view_ind_dep),
                       np.sum(cv_score_view_ind_dep_cont)]
            cv_trial = pd.DataFrame(data = df_data, index=df_index)
     
            total_results[vpn] = cv_trial
        debug = {'error?': (cv_score_view_ind_dep_cont,df_data,m_6),
                 'hold':holdout_data,
                 'test':test_data,
                 'suppl': (test_L,hold_L)}
        return (total_results, debug)        


        
        
# #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
# self.VPN_outputs = np.array(data['answer']) 
# self.intro_newID = np.array(data['new_IDs']) 
# self.n_prev_VI = np.array(curr_data_vpn['n_prev_VI'])
# self.n_prev_pres_VD = 
# self.stim_IDs_VI=    np.array(curr_data_vpn['stim_IDs_VI'])  #stimulus IDs of winning model 
# new_ID=         np.array(curr_data_vpn['new_IDs'])      #trials where new ID is introduced 
# n_prev_pres=        #number_of_prev_presentations
# stim_IDs_VD=    np.array(curr_data_vpn['stim_IDs_VD'])  #view dependent
# VPN_output =          #VPN answers
# verbose =       False
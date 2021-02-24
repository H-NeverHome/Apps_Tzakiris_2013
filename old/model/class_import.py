# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:45:32 2020

@author: de_hauk
"""

########### Fetch Data Funcs

def get_data(path):
    
    '''old & deprec'''
    import pandas as pd
    ### Get preprocessed and previously stored data 
    data = pd.read_csv(path + r'/final_proc_dat_labjansen.csv')

    #### get unique IDS
    sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
    sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]
    
    return {'data':data,
            'view_ind_data':sample_answer_clms,
            'view_dep_data':sample_perspective_clms}

def get_data_3(path_raw_data, ground_truth_file):
    ### Process raw data under path, nothin prestored
    
    import pandas as pd
    import glob
    import numpy as np
    from autoimpute.imputations import MultipleImputer
    data_path = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
    ground_truth_file = r'C:\Users\de_hauk\HESSENBOX\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv'
    all_files = glob.glob(data_path + "/*.csv")
    
    SAMPLE_fullinfo = pd.read_csv(ground_truth_file).drop(columns = ['Unnamed: 0']).copy()  
    
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
    return final_dat

def get_data_2(path_raw_data, ground_truth_file):
    ### Process raw data under path, nothin prestored
    
    import pandas as pd
    import glob
    import numpy as np
    from autoimpute.imputations import MultipleImputer
    data_path = path_raw_data
    #data_path = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
    all_files = glob.glob(data_path + "/*.csv")
    
    sample_fullinfo = pd.read_csv(ground_truth_file)
    SAMPLE_fullinfo = sample_fullinfo.drop(columns = ['Unnamed: 0']).copy()  
    
    unq_id = []
    DATA_raw_DF = pd.DataFrame()
    for data in all_files:
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
                
    
        DATA_raw_DF[unique_ID + '_perspective'] = perspective
        DATA_raw_DF[unique_ID + '_perf'] = answer_correct
        DATA_raw_DF[unique_ID + '_answer'] = answer_raw
    #raw_dat_unproc = [pd.read_csv(i,header=None, sep='\t').drop(axis='index', labels = [0,1])[2] for i in all_files]
    
    ### determine place and amount of missing value
    missing_dat_raw = pd.DataFrame(DATA_raw_DF.isnull().sum(), columns = ['dat'])
    
    ### filter out clmns with no missing dat
    #missing_dat_overview = missing_dat_raw.loc[missing_dat_raw['dat'] > 0].T
    
    #drop irrelevant columns containig weird stim codings
    miss_perspect = [i for i in DATA_raw_DF if ('perspective' in i) or ('perf' in i)]
    miss_dat = DATA_raw_DF.drop(labels =miss_perspect, axis = 1 ).copy()
    
    # Impute missing dat with binary logistic regress returning only one DF with imputed values
    ### Uses = https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    imp = MultipleImputer(n=1,strategy= 'binary logistic',return_list=True, imp_kwgs={"binary logistic": {"max_iter": 10000}} )
    impute_res = imp.fit_transform(miss_dat)
    
    # # merge imputed DF with relevant Info i.e. generating list etc
    DATA_imput = impute_res[0][1]
    for i in SAMPLE_fullinfo:
            DATA_imput[i] = sample_fullinfo[i]
    for i in DATA_raw_DF:
        if i not in [i for i in DATA_imput]:
            DATA_imput[i] = DATA_raw_DF[i]
    return {'imputed_data':DATA_imput,
            'raw_data': DATA_raw_DF,
            'stat_ground-truth':SAMPLE_fullinfo,
            'unique_ID': unq_id}

def data_old_sample(path):
    '''old & deprec'''
    import pandas as pd
    import glob
    import numpy as np
    from autoimpute.imputations import MultipleImputer
    
    data_path = path
    
    
    ############################################### Data #################################################################################
    
    ''' Data Processing in two batches. Second batch containes additional Information in form of an additional row per trial which 
    needs to be deleted'''
    
    ## First Batch
    path_1_batch_raw = data_path + r'\first_batch'
    all_files_1B = glob.glob(path_1_batch_raw + "/*.csv")
    
    ## Second Batch
    path_2_batch_raw = data_path + r'\second_batch'
    all_files_2B = glob.glob(path_2_batch_raw + "/*.csv")
    
    ## Merge Batches so that subj 1-5 are first batch and 6-10 are second batch
    all_files = all_files_1B + all_files_2B
    
    ## Get raw data i.e. trial list generated by previous optimization run
    sample_fullinfo = pd.read_csv(data_path + r'\stimuli_list.csv')
    sample_fullinfo = sample_fullinfo.drop(columns = ['Unnamed: 0']).copy()
    
    ## setup dataframe not containing raw dat
    SAMPLE_onlydat = pd.DataFrame()
    
    raw_dat_unproc = [pd.read_csv(i,sep='\s+', header=None).drop(columns=[0,1,2,5,6,7,8,9]).drop(axis='index', labels = [0]) for i in all_files]
    
    ## For each (subject)-data in directories
    for ID in all_files:
        
        curr_ID = ID[-8:-6] ## get unique ID Number/Subject Number from raw dat names
        curr_file = ID ## get current file-path
        
        ## get actual data into dataframe & drop irrelevant clmns
        df_sample_raw = pd.read_csv(curr_file,sep='\s+', header=None).drop(columns=[0,1,2,5,6,7,8,9]).drop(axis='index', labels = [0])
        
        ## select only relevant rows
        ## ACHTUNG: Length of df = batch 1 < batch 2
        ## length use as a seperator to filter irrelevant rows
        
        if len(df_sample_raw[3]) >= 800: # aka. if data from batch 2
            df_sample = df_sample_raw.loc[(df_sample_raw[4] != '2=switched_1=notswitched') & (df_sample_raw[4] != 'actual_key_press_for_RT')]
        else:
            df_sample = df_sample_raw.loc[(df_sample_raw[4] != 'actual_key_press_for_RT')]
            
        ## Initialize protocoll data
        answer_correct = [] #1==yes//0==no 
        faceID_known = [] #1==yes//0==no
        perspective = []
        
        for i,j in zip(df_sample[3], df_sample[4]):
    
            ### Which stimulus was observed
            if (len(i) == 4) and (i not in ['None','right','left']):
                perspective.append(i)
                
            ### What was the Answer, regardless of correctness??  
            if ('right' in str(i)) and (str(j) == 'left=yes_right=no_None=missed'):
                faceID_known.append(0) # NOT familiar
            elif ('left' in str(i)) and (str(j) == 'left=yes_right=no_None=missed'):
                faceID_known.append(1) # familiar
            elif (str(i) == 'None') and (str(j) == 'left=yes_right=no_None=missed'):
                faceID_known.append(np.nan) # code responding too early and missed responses as np.nan
                
            ### Was the answer correct w.r.t. to the task? // Missing an Answer is also worng
            if (str(i) == '1') and ( 'right' in str(j)):
                answer_correct.append(int(i))
            elif (str(i) == '0') and ( 'wrong' in str(j)):
                answer_correct.append(int(i))
            elif (str(i) == '0') and (str(j) == 'missed'):
                answer_correct.append(int(i))           
          
        
        sample_fullinfo[str(curr_ID) + 'answer'] = pd.Series(faceID_known)    
        sample_fullinfo[str(curr_ID) + 'perf'] = pd.Series(answer_correct)
        sample_fullinfo[str(curr_ID) + 'perspective'] = pd.Series(perspective)
        
        SAMPLE_onlydat[str(curr_ID) + 'answer'] = pd.Series(faceID_known)
        SAMPLE_onlydat[str(curr_ID) + 'perf'] = pd.Series(answer_correct)
        SAMPLE_onlydat[str(curr_ID) + 'perspective'] = pd.Series(perspective)
    
    
    #################################### Impute missing Data ###############################################################
    
    ### determine place and amount of missing value
    missing_dat_raw = pd.DataFrame(sample_fullinfo.isnull().sum(), columns = ['dat'])
    
    ### filter out clmns with no missing dat
    missing_dat_overview = missing_dat_raw.loc[missing_dat_raw['dat'] > 0].T
    
    
    
    #drop irrelevant columns containig weird stim codings
    miss_perspect = [i for i in SAMPLE_onlydat if 'perspective' in i ]
    miss_dat = SAMPLE_onlydat.drop(labels =miss_perspect, axis = 1 ).copy()
    
    # Impute missing dat with binary logistic regress returning only one DF with imputed values
    ### Uses = https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    imp = MultipleImputer(n=1,strategy= 'binary logistic',return_list=True, imp_kwgs={"binary logistic": {"max_iter": 10000}} )
    impute_res = imp.fit_transform(miss_dat)
    
    # merge imputed DF with relevant Info i.e. generating list etc
    DATA_imput = impute_res[0][1]
    for i in sample_fullinfo:
        if i not in [i for i in sample_fullinfo if ('answer' in i) or ('perf' in i)]:
            DATA_imput[i] = sample_fullinfo[i]
      
    return {'Data_raw_unproc':raw_dat_unproc,
            'Data_raw': SAMPLE_onlydat,
            'DATA_imput': DATA_imput,
            }

def reformat_data_within_T(data_2_sample):
    ref_dat_raw = data_2_sample['raw_data']
    unq_ids = data_2_sample['unique_ID']
    
    
    data_dict_total = {}
    data_dict_t1 = {}
    data_dict_t2 = {}
    for ids in unq_ids:
        curr_df_raw = ref_dat_raw[[i for i in ref_dat_raw if ids in i]].copy()
        curr_df_raw['stim_IDs'] = list(data_2_sample['stat_ground-truth']['stim_IDs'])
        curr_df_raw['new_IDs'] = list(data_2_sample['stat_ground-truth']['new_IDs'])
        curr_df_raw['n_prev_pres'] = list(data_2_sample['stat_ground-truth']['number_of_prev_presentations_raw '])
        
        curr_df = curr_df_raw.loc[curr_df_raw[ids+'_answer'].isna() == False].copy()
        curr_df_1 = curr_df.reset_index(drop=True, inplace=False)
        data_dict_total[ids] = curr_df_1
        if 'A' in ids:
            data_dict_t1[ids] = curr_df_1
        elif 'B' in ids:
            data_dict_t2[ids] = curr_df_1
    return ('data_dict_t1',data_dict_t1,'data_dict_t2',data_dict_t2)


########### Task & Reliability Funcs 

def get_behavioral_performance(subject_data):
    import pandas as pd
    data = subject_data
    ########## Behavioral Performance only raw_dat

    vpn_ids = [i.split('_a')[0] for i in data['raw_data'] if 'answer' in i]
    vpn_answer = [i for i in data['raw_data'] if 'answer' in i]
    vpn_perf = [i.split('_a')[0] for i in data['raw_data'] if '_perf' in i]
    behavioral_perf = pd.DataFrame(index = vpn_ids)
    
    ##### Get %-correct
    perf_dat = [(data['raw_data'][i].copy().sum()/ len(data['raw_data'][i]))*100 for i in vpn_perf]
    behavioral_perf['%-correct'] = perf_dat
    
    ##### Get n_errors
    behavioral_perf['n_errors'] = [len(data['raw_data'][i])-data['raw_data'][i].sum() for i in vpn_perf]
    
    ##### Get Missed
    
    behavioral_perf['missed_answers'] = [data['raw_data'][i].isna().sum() for i in data['raw_data'] if 'answer' in i]
    
    behavioral_perf_group = pd.DataFrame(index = [i + '_M' for i in behavioral_perf.copy()] + [i + '_SD' for i in behavioral_perf.copy()] )
    
    
    data_g_A_M = behavioral_perf.copy().loc[[i for i in behavioral_perf.index if 'A' in i]].mean(axis = 0)
    data_g_B_M = behavioral_perf.copy().loc[[i for i in behavioral_perf.index if 'B' in i]].mean(axis = 0)
    data_g_A_SD = behavioral_perf.copy().loc[[i for i in behavioral_perf.index if 'A' in i]].std(axis = 0)
    data_g_B_SD = behavioral_perf.copy().loc[[i for i in behavioral_perf.index if 'B' in i]].std(axis = 0)
        
    
    behavioral_perf_group['A'] = [i for i in data_g_A_M] + [i for i in data_g_A_SD]
    behavioral_perf_group['B'] = [i for i in data_g_B_M] + [i for i in data_g_B_SD]
    final_dict = {'behavioral_results_subj': behavioral_perf,
                  'behavioral_results_group': behavioral_perf_group.T,
                  'used_data':data['raw_data']}
    return final_dict

def task_icc(task_perf):
    import pingouin as pg
    import pandas as pd
    icc_data = task_perf['behavioral_results_subj'].copy()
    icc_data['time'] = [i[-1] for i in icc_data.index]
    icc_data['id'] = [i[0] for i in icc_data.index]

    icc = pg.intraclass_corr(data=icc_data, 
                         targets='id', 
                         raters='time',
                         ratings='%-correct').round(3)
    return icc.loc[icc['Type'] == 'ICC2']

def pears_corr(task_perf):
    import pingouin as pg
    import pandas as pd
    corr_data = task_perf['behavioral_results_subj'].copy()
    corr_data['time'] = [i[-1] for i in corr_data.index]
    corr_data_A = corr_data.loc[corr_data['time'] == 'A']['%-correct']
    corr_data_B = corr_data.loc[corr_data['time'] == 'B']['%-correct']
    corr = pg.corr(corr_data_A, corr_data_B, tail='two-sided', method='pearson')
    return corr

def task_rel(task_perf):
    corr = pears_corr(task_perf)
    icc = task_icc(task_perf)
    return (corr,icc,task_perf)
    
    

########### Data Fitting Funcs

def fit_data_noCV(data, lbfgs_epsilon, verbose_tot):
        
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    
    
    
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    np.random.seed(1993)
    ### Get data 
    data = data
    
    #### get unique IDS
    sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
    sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]
    
    epsilon_param = lbfgs_epsilon
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
                    
    data_verbose_debug = {}
    res_evidence = pd.DataFrame(index=models_names)
    trialwise_data = {}
    bf_log_group = pd.DataFrame()
    
    for i,j in zip(sample_answer_clms,sample_perspective_clms):
        print(i)
        #func calls & rand starts
    
        # data import
        stim_IDs = data['stim_IDs'] #stimulus IDs of winning model 
        new_ID = data['new_IDs'] #trials where new ID is introduced 
        numb_prev_presentations = data['number_of_prev_presentations_raw '] #number_of_prev_presentations
        stim_IDs_perspective = data[str(j)] #view dependent
        VPN_output = data[str(i)] #VPN answers
        verbose = False
        
        
        ##### Model Optim
        
        
        print('VIEW_INDIPENDENTxCONTEXT')
        data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

        bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
        
        part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
        res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
        
        
        print('VIEW_DEPENDENT')
        
        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]

    
        bounds_M2 = [(0,1),(.1,20),(0,2)]
        
        part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
        res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                      approx_grad = True,
                                      bounds = bounds_M2, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                      epsilon=epsilon_param)
        
        parameter_est['VIEW_DEPENDENT'][i] = res2[0]
        
        
        print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        
        data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
    
        part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
        res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                      approx_grad = True,
                                      bounds = bounds_M3, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
        
        print('VIEW_INDEPENDENT')

        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
        data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
        bounds_M4 = [(0,1),(.1,20),(0,2)]
    
        part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
        res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                      approx_grad = True,
                                      bounds = bounds_M4, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
        
        
        print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
        
        #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
    
        data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
        res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                      approx_grad = True,
                                      bounds = bounds_M5, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
        

        print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
        params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
        data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
        res6 = optimize.fmin_l_bfgs_b(part_func_M6,
                                      approx_grad = True,
                                      bounds = bounds_M6, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M6))],
                                      epsilon=epsilon_param)
        
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT'][i] = res6[0]
        
        re_evidence_subj = np.array([(-1*i[1]) for i in [res1,res2,res3,res4,res5,res6]])
        res_evidence[i] = re_evidence_subj
        
        ### Subject BF_LOG
        bf_log_subj = re_evidence_subj[0]-special.logsumexp(np.array(re_evidence_subj[1::]))
        bf_log_group[i + '_BF_log'] = [bf_log_subj]
        
        
        # trialwise_dat = {}
        ############################## Verbose == True ###########################
        
        verbose_debug = verbose_tot
        
        data_M_debug = [data_M1, data_M2, data_M3, data_M4, data_M5, data_M6]
        for dat in data_M_debug:
            dat[-1] = True        
        
        if verbose_debug == True:

            data_M1_debug = data_M_debug[0]
            params_m_1 = res1[0]
            m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
            
            data_M2_debug = data_M_debug[1]
            params_m_2 = res2[0]
            m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
        
            data_M3_debug = data_M_debug[2]
            params_m_3 = res3[0]
            m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
            
            data_M4_debug = data_M_debug[3]
            params_m_4 = res4[0]
            m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
        
            data_M5_debug = data_M_debug[4]
            params_m_5 = res5[0]
            m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
            
            data_M6_debug = data_M_debug[5]
            params_m_6 = res6[0]
            m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)
    
            res_debug = {models_names[0]: m_1,
                         models_names[1]: m_2,
                         models_names[2]: m_3,
                         models_names[3]: m_4,
                         models_names[4]: m_5,
                         models_names[5]: m_6}
            data_verbose_debug[i] = res_debug
            
    #### Get winning model trialwise dat ####
        data_M1_debug = data_M_debug[0]
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)        

        trialwise_data[i] = m_1[1]['data_store_1']
    restotal = res_evidence.sum(axis=1)
    cntrl_log = special.logsumexp(np.array(restotal[1::]))
    bf_log = (cntrl_log -(np.array(restotal[0])))
    if verbose_tot==True:
        return (restotal,data_verbose_debug)
    elif verbose_tot==False:
        return {'uncorr_LR_10':np.exp(-1*bf_log),
                'subject_level_model_evidence':res_evidence,
                'group_level_model_evidence':res_evidence.sum(axis=1),
                'subject_level_uncorr_LR': bf_log_group,
                #'total_model-evidence':restotal,
                'used_data': data,
                'subject_level_parameter-estimates':parameter_est,
                'subject_level_trialwise_data_win_model':trialwise_data}
                
def fit_data_noCV_irr_len_data(data, lbfgs_epsilon, verbose_tot):
        
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    
    
    
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    np.random.seed(1993)
    ### Get data 
    data = data
    
    #### get unique IDS
    unique_id = list(data.keys())
    sample_answer_clms = [i+'_answer' for i in unique_id]
    sample_perspective_clms = [i+'_perspective' for i in unique_id]
    
    epsilon_param = lbfgs_epsilon
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
        print(vpn)
        #func calls & rand starts
        
        curr_data_vpn = data[vpn]
        # data import
        stim_IDs = curr_data_vpn['stim_IDs'] #stimulus IDs of winning model 
        new_ID = curr_data_vpn['new_IDs'] #trials where new ID is introduced 
        numb_prev_presentations = curr_data_vpn['n_prev_pres'] #number_of_prev_presentations
        stim_IDs_perspective = curr_data_vpn[vpn+'_perspective'] #view dependent
        VPN_output = curr_data_vpn[vpn+'_answer'] #VPN answers
        verbose = False
        
        
        ##### Model Optim
        
        i=vpn
        print('VIEW_INDIPENDENTxCONTEXT')
        data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

        bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
        
        part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
        res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
        
        
        print('VIEW_DEPENDENT')
        
        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]

    
        bounds_M2 = [(0,1),(.1,20),(0,2)]
        
        part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
        res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                      approx_grad = True,
                                      bounds = bounds_M2, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                      epsilon=epsilon_param)
        
        parameter_est['VIEW_DEPENDENT'][i] = res2[0]
        
        
        print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        
        data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
    
        part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
        res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                      approx_grad = True,
                                      bounds = bounds_M3, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
        
        print('VIEW_INDEPENDENT')

        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
        data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
        bounds_M4 = [(0,1),(.1,20),(0,2)]
    
        part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
        res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                      approx_grad = True,
                                      bounds = bounds_M4, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
        
        
        print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
        
        #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
    
        data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
        res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                      approx_grad = True,
                                      bounds = bounds_M5, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
        

        print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
        params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
        data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
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
        
        data_M_debug = [data_M1, data_M2, data_M3, data_M4, data_M5, data_M6]
        for dat in data_M_debug:
            dat[-1] = True        
        
        if verbose_debug == True:

            data_M1_debug = data_M_debug[0]
            params_m_1 = res1[0]
            m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
            
            data_M2_debug = data_M_debug[1]
            params_m_2 = res2[0]
            m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
        
            data_M3_debug = data_M_debug[2]
            params_m_3 = res3[0]
            m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
            
            data_M4_debug = data_M_debug[3]
            params_m_4 = res4[0]
            m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
        
            data_M5_debug = data_M_debug[4]
            params_m_5 = res5[0]
            m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
            
            data_M6_debug = data_M_debug[5]
            params_m_6 = res6[0]
            m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)
    
            res_debug = {models_names[0]: m_1,
                         models_names[1]: m_2,
                         models_names[2]: m_3,
                         models_names[3]: m_4,
                         models_names[4]: m_5,
                         models_names[5]: m_6}
            data_verbose_debug[i] = res_debug
            
    #### Get winning model trialwise dat ####
        data_M1_debug = data_M_debug[0]
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)        

        trialwise_data[i] = m_1[1]['data_store_1']
    restotal = res_evidence.sum(axis=1)
    cntrl_log = special.logsumexp(np.array(restotal[1::]))
    bf_log = (cntrl_log -(np.array(restotal[0])))
    
    results_1 = {'uncorr_LR_10':np.exp(-1*bf_log),
                'subject_level_model_evidence':res_evidence,
                'group_level_model_evidence':res_evidence.sum(axis=1),
                'subject_level_uncorr_LR': bf_log_group,
                'xxx':data_verbose_debug,
                'used_data': data,
                'subject_level_parameter-estimates':parameter_est,
                'subject_level_trialwise_data_win_model':trialwise_data,
                'baseline_model': np.sum(np.log([0.5 for i in range(189)]))}
    if verbose_tot==True:
        return (results_1,restotal,data_verbose_debug)
    elif verbose_tot==False:
        return results_1

def data_fit_t1_t2_comb(data_t1,data_t2, lbfgs_epsilon):
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    np.random.seed(1993)

    #### get unique IDS
    unique_id_t1 = list(data_t1.keys())
    unique_id_t2 = list(data_t2.keys())
    
    sample_answer_clms_t1 = [i+'_answer' for i in unique_id_t1]
    sample_answer_clms_t2 = [i+'_answer' for i in unique_id_t2]
    
    sample_perspective_clms = [i+'_perspective' for i in unique_id_t1]
    sample_perspective_clms = [i+'_perspective' for i in unique_id_t2]
    
    epsilon_param = lbfgs_epsilon
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
        
        curr_data_vpn_t1 = data_t1[vpn_t1]
        curr_data_vpn_t2 = data_t2[vpn_t2]
        # data import
        stim_ID_t1 = curr_data_vpn_t1['stim_IDs'] #stimulus IDs of winning model
        stim_ID_t2 = curr_data_vpn_t2['stim_IDs'] #stimulus IDs of winning model
        
        new_ID_t1 = curr_data_vpn_t1['new_IDs'] #trials where new ID is introduced 
        new_ID_t2 = curr_data_vpn_t2['new_IDs'] #trials where new ID is introduced 
        
        numb_prev_presentations_t1 = curr_data_vpn_t1['n_prev_pres'] #number_of_prev_presentations
        numb_prev_presentations_t2 = curr_data_vpn_t2['n_prev_pres'] #number_of_prev_presentations

        stim_IDs_perspective_t1 = curr_data_vpn_t1[vpn_t1+'_perspective'] #view dependent
        stim_IDs_perspective_t2 = curr_data_vpn_t2[vpn_t2+'_perspective'] #view dependent

        VPN_output_t1 = curr_data_vpn_t1[vpn_t1+'_answer'] #VPN answers
        VPN_output_t2  = curr_data_vpn_t2[vpn_t2+'_answer'] #VPN answers
        verbose = False


        #################### Model Optim      
        #i=vpn
####################  VIEW_INDIPENDENTxCONTEXT     ####################        
        def VIEW_INDIPENDENTxCONTEXT_t1_t2(data_M1_t1,data_M1_t2,params):
           res1_t1 = VIEW_INDIPENDENTxCONTEXT(data_M1_t1,params)
           res1_t2 = VIEW_INDIPENDENTxCONTEXT(data_M1_t2,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss
       
        data_M1_t1 = [VPN_output_t1, new_ID_t1, numb_prev_presentations_t1, stim_ID_t1,verbose]
        data_M1_t2 = [VPN_output_t2, new_ID_t2, numb_prev_presentations_t2, stim_ID_t2,verbose]
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
           res1_t1 = VIEW_DEPENDENT(data_M2_t1,params)
           res1_t2 = VIEW_DEPENDENT(data_M2_t1,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss
        
        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        data_M2_t1 = [VPN_output_t1, 
                      new_ID_t1, 
                      stim_ID_t1,
                      stim_IDs_perspective_t1,
                      verbose]
        data_M2_t2 = [VPN_output_t2, 
                      new_ID_t2, 
                      stim_ID_t2,
                      stim_IDs_perspective_t2,
                      verbose]

    
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
           res1_t1 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_t1,params)
           res1_t2 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_t2,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss
        data_M3_t1 = [VPN_output_t1, 
                      new_ID_t1, 
                      stim_ID_t1,
                      stim_IDs_perspective_t1,
                      verbose]
        data_M3_t2 = [VPN_output_t2, 
                      new_ID_t2, 
                      stim_ID_t2,
                      stim_IDs_perspective_t2,
                      verbose]
 
        
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
           res1_t1 = VIEW_INDEPENDENT(data_M4_t1,params)
           res1_t2 = VIEW_INDEPENDENT(data_M4_t2,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss

        data_M4_t1 = [VPN_output_t1, 
                      new_ID_t1,
                      numb_prev_presentations_t1,
                      stim_ID_t1,
                      verbose]
        data_M4_t2 = [VPN_output_t2, 
                      new_ID_t2,
                      numb_prev_presentations_t2,
                      stim_ID_t2,
                      verbose]

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
           res1_t1 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_t1,params)
           res1_t2 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_t2,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss

        data_M5_t1 = [VPN_output_t1, 
                      new_ID_t1,
                      numb_prev_presentations_t1,
                      stim_ID_t1,
                      stim_IDs_perspective_t1,
                      verbose]
        data_M5_t2 = [VPN_output_t2, 
                      new_ID_t2,
                      numb_prev_presentations_t2,
                      stim_ID_t2,
                      stim_IDs_perspective_t2,
                      verbose]

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
           res1_t1 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_t1,params)
           res1_t2 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_t2,params)                     
           loss_t1 = res1_t1
           loss_t2 = res1_t2
           total_loss = loss_t1 + loss_t2
           return total_loss

        data_M6_t1 = [VPN_output_t1, 
                      new_ID_t1,
                      numb_prev_presentations_t1,
                      stim_ID_t1,
                      stim_IDs_perspective_t1,
                      verbose]
        data_M6_t2 = [VPN_output_t2, 
                      new_ID_t2,
                      numb_prev_presentations_t2,
                      stim_ID_t2,
                      stim_IDs_perspective_t2,
                      verbose]

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
           
    
    
    
    return res_evidence



def fit_data_CV(data, lbfgs_epsilon, verbose_tot):   
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT_CV,VIEW_DEPENDENT_CV
    from model_functions_BFGS import VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV,VIEW_INDEPENDENT_CV
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENT_CV
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV
    #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    
    
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    np.random.seed(1993)
    ### Get data 
    data = data
    
    #### get unique IDS
    unique_id = list(data.keys())
    sample_answer_clms = [i+'_answer' for i in unique_id]
    sample_perspective_clms = [i+'_perspective' for i in unique_id]
    
    epsilon_param = lbfgs_epsilon
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
    
    
    for vpn in unique_id[1::]:
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


def fit_data_CV_joblib(data, lbfgs_epsilon, verbose_tot):
        
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    
    
    
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize,special
    np.random.seed(1993)
    ### Get data 
    data = data
    
    #### get unique IDS
    unique_id = list(data.keys())
    sample_answer_clms = [i+'_answer' for i in unique_id]
    sample_perspective_clms = [i+'_perspective' for i in unique_id]
    
    epsilon_param = lbfgs_epsilon
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
        print(vpn)
        #func calls & rand starts
        
        curr_data_vpn = data[vpn]
        # data import
        stim_IDs = curr_data_vpn['stim_IDs'] #stimulus IDs of winning model 
        new_ID = curr_data_vpn['new_IDs'] #trials where new ID is introduced 
        numb_prev_presentations = curr_data_vpn['n_prev_pres'] #number_of_prev_presentations
        stim_IDs_perspective = curr_data_vpn[vpn+'_perspective'] #view dependent
        VPN_output = curr_data_vpn[vpn+'_answer'] #VPN answers
        verbose = False
        
        
        ##### Model Optim
        
        i=vpn
        print('VIEW_INDIPENDENTxCONTEXT')
        data_M1 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]

        bounds_M1 = [(0,1),(0,1),(.1,20),(0,2)]
        
        part_func_M1 = partial(VIEW_INDIPENDENTxCONTEXT,data_M1) 
        res1 = optimize.fmin_l_bfgs_b(part_func_M1,
                                      approx_grad = True,
                                      bounds = bounds_M1, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M1))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDIPENDENTxCONTEXT'][i] = res1[0]
        
        
        print('VIEW_DEPENDENT')
        
        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        data_M2 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]

    
        bounds_M2 = [(0,1),(.1,20),(0,2)]
        
        part_func_M2 = partial(VIEW_DEPENDENT,data_M2) 
        res2 = optimize.fmin_l_bfgs_b(part_func_M2,
                                      approx_grad = True,
                                      bounds = bounds_M2, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M2))],
                                      epsilon=epsilon_param)
        
        parameter_est['VIEW_DEPENDENT'][i] = res2[0]
        
        
        print('VIEW_DEPENDENTxCONTEXT_DEPENDENT')

        #data = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]    
        
        data_M3 = [VPN_output, new_ID, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M3 = [(0,1),(0,1),(.1,20),(0,2)]
    
        part_func_M3 = partial(VIEW_DEPENDENTxCONTEXT_DEPENDENT,data_M3) 
        res3 = optimize.fmin_l_bfgs_b(part_func_M3,
                                      approx_grad = True,
                                      bounds = bounds_M3, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M3))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_DEPENDENTxCONTEXT_DEPENDENT'][i] = res3[0]
        
        print('VIEW_INDEPENDENT')

        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
    
        data_M4 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs,verbose]
        bounds_M4 = [(0,1),(.1,20),(0,2)]
    
        part_func_M4 = partial(VIEW_INDEPENDENT,data_M4) 
        res4 = optimize.fmin_l_bfgs_b(part_func_M4,
                                      approx_grad = True,
                                      bounds = bounds_M4, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M4))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENT'][i] = res4[0]
        
        
        print('VIEW_INDEPENDENTxVIEW_DEPENDENT')
        
        #data = [ VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
    
        data_M5 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M5 = [(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M5 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENT,data_M5) 
        res5 = optimize.fmin_l_bfgs_b(part_func_M5,
                                      approx_grad = True,
                                      bounds = bounds_M5, 
                                      x0 = [x_0_bfgs for i in range(len(bounds_M5))],
                                      epsilon=epsilon_param)
        parameter_est['VIEW_INDEPENDENTxVIEW_DEPENDENT'][i] = res5[0]
        

        print('VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT')
        params_M6_name = ['alpha_ind', 'alpha_dep', 'sigma', 'beta', 'lamd_a_ind', 'lamd_a_dep']
        #data = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
    
        data_M6 = [VPN_output, new_ID, numb_prev_presentations, stim_IDs, stim_IDs_perspective, verbose]
        bounds_M6 = [(0,1),(0,1),(0,1),(.1,20),(0,2),(0,2)]
    
        part_func_M6 = partial(VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT,data_M6) 
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
        
        data_M_debug = [data_M1, data_M2, data_M3, data_M4, data_M5, data_M6]
        for dat in data_M_debug:
            dat[-1] = True        
        
        if verbose_debug == True:

            data_M1_debug = data_M_debug[0]
            params_m_1 = res1[0]
            m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
            
            data_M2_debug = data_M_debug[1]
            params_m_2 = res2[0]
            m_2 = VIEW_DEPENDENT(data_M2_debug, params_m_2)
        
            data_M3_debug = data_M_debug[2]
            params_m_3 = res3[0]
            m_3 = VIEW_DEPENDENTxCONTEXT_DEPENDENT(data_M3_debug, params_m_3)
            
            data_M4_debug = data_M_debug[3]
            params_m_4 = res4[0]
            m_4 = VIEW_INDEPENDENT(data_M4_debug, params_m_4)
        
            data_M5_debug = data_M_debug[4]
            params_m_5 = res5[0]
            m_5 = VIEW_INDEPENDENTxVIEW_DEPENDENT(data_M5_debug, params_m_5)
            
            data_M6_debug = data_M_debug[5]
            params_m_6 = res6[0]
            m_6 = VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT(data_M6_debug, params_m_6)
    
            res_debug = {models_names[0]: m_1,
                         models_names[1]: m_2,
                         models_names[2]: m_3,
                         models_names[3]: m_4,
                         models_names[4]: m_5,
                         models_names[5]: m_6}
            data_verbose_debug[i] = res_debug
            
    #### Get winning model trialwise dat ####
        data_M1_debug = data_M_debug[0]
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)        

        trialwise_data[i] = m_1[1]['data_store_1']
    restotal = res_evidence.sum(axis=1)
    cntrl_log = special.logsumexp(np.array(restotal[1::]))
    bf_log = (cntrl_log -(np.array(restotal[0])))
    
    results_1 = {'uncorr_LR_10':np.exp(-1*bf_log),
                'subject_level_model_evidence':res_evidence,
                'group_level_model_evidence':res_evidence.sum(axis=1),
                'subject_level_uncorr_LR': bf_log_group,
                'xxx':data_verbose_debug,
                'used_data': data,
                'subject_level_parameter-estimates':parameter_est,
                'subject_level_trialwise_data_win_model':trialwise_data,
                'baseline_model': np.sum(np.log([0.5 for i in range(189)]))}
    if verbose_tot==True:
        return (results_1,restotal,data_verbose_debug)
    elif verbose_tot==False:
        return results_1

########### Model Selection Funcs
   
def model_selection_AT():
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

def bayes_RFX_cond(fit_data_sample_T1,fit_data_sample_T2):
    import matlab.engine
    import numpy as np
    eng = matlab.engine.start_matlab()
    #see https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/#between-conditions-rfx-bms
    a_t = fit_data_sample_T1['subject_level_model_evidence'].copy()
    b_t = fit_data_sample_T2['subject_level_model_evidence'].copy()
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

def orig_procedure(fit_data_sample_T1, fit_data_sample_T2):
    import pingouin as pg
    import numpy as np
    import pandas as pd
    ind_T1 = fit_data_sample_T1['subject_level_model_evidence'].copy().T
    ind_T2 = fit_data_sample_T2['subject_level_model_evidence'].copy().T
    model_names = [i for i in fit_data_sample_T1['subject_level_model_evidence'].index]
    
    res_tt = pd.DataFrame(index=['p_val','d','BF10','power'])
    for sample,name_s in zip([ind_T1,ind_T2],['_A','_B']):
        for model in model_names[1::]:
            data_W = sample[model_names[0]]
            data_C = sample[model]
            res_tt_ind_raw = pg.ttest(data_W,data_C,False,'two-sided')
            res_tt_ind = [res_tt_ind_raw['p-val'][0],
                          res_tt_ind_raw['cohen-d'][0],
                          res_tt_ind_raw['BF10'][0],
                          res_tt_ind_raw['power'][0]]
            name_res = str(model_names[0]) + '_vs_' + str(model) + name_s
            res_tt[name_res] = [float(i) for i in res_tt_ind]
    res_tt = res_tt.copy()
    pval_A = [res_tt[i][0] for i in res_tt if 'A' in i]
    pval_B = [res_tt[i][0] for i in res_tt if 'B' in i]
    pval_A_FDR = pg.multicomp(pval_A, alpha=0.02, method='fdr_bh')[1]
    pval_B_FDR = pg.multicomp(pval_B, alpha=0.02, method='fdr_bh')[1]
    res_tt = res_tt.copy().T.sort_index()
    res_tt['time'] = [i[-1] for i in res_tt.index]
    #ress = [i for i in res_tt.index]
    res_tt = res_tt.copy().sort_values('time',axis='index')
    
    # see https://www.stat.cmu.edu/~genovese/talks/hannover1-04.pdf
    res_tt['p_val_BH_FDR'] = list(pval_A_FDR) + list(pval_B_FDR)
    return (res_tt,pval_A,pval_B)

def bic_modelselect(fit_data_sample_T1, fit_data_sample_T2):
    import pingouin as pg
    import numpy as np
    import pandas as pd
    name_models_bic = [i for i in fit_data_sample_T1['subject_level_parameter-estimates']]
    n_params = pd.DataFrame()
    for model in name_models_bic:
        n_par = fit_data_sample_T1['subject_level_parameter-estimates'][model].shape[0]
        n_params[model] = [n_par]
    
    def bic(LL,n_param,n):
        bic = [int((n_param*np.log(n))-(2*i)) for i in LL]
        return bic
    
    bic_inx_A = [i for i in fit_data_sample_T1['subject_level_model_evidence'].copy().T.index]
    bic_inx_B = [i for i in fit_data_sample_T2['subject_level_model_evidence'].copy().T.index]
    
    res_bic = pd.DataFrame(index= bic_inx_A + bic_inx_B)
    res_bic_D = pd.DataFrame(index = [i[0] for i in bic_inx_A])
    for i in name_models_bic:
        raw_LL_A = fit_data_sample_T1['subject_level_model_evidence'].copy().T[i]
        raw_LL_B = fit_data_sample_T2['subject_level_model_evidence'].copy().T[i]
         
        n_size_bic = len(raw_LL_B)
        n_param = n_params[i]
        bic_list_A = bic(raw_LL_A,n_param,n_size_bic)
        bic_list_B = bic(raw_LL_B,n_param,n_size_bic)
        res_bic[i] = bic_list_A + bic_list_B
        delta_bic = [bic_A - bic_B for bic_A,bic_B in zip(bic_list_A, bic_list_B)]
        #delta_bic_norm = [(x-y)/(x+y) for bic_A,bic_B in zip(bic_list_A, bic_list_B)]
        res_bic_D[i] = delta_bic
        #res_bic_D[i+ '_norm'] = delta_bic_norm
    return {'subject_wise_BIC' : res_bic,
            'diff_BIC_(A-B)':res_bic_D}


def corr_lr_func(fit_data):
    import pandas as pd
    import numpy as np
    name_models_LR = [i for i in fit_data['subject_level_parameter-estimates']]
    n_params = pd.DataFrame()
    for i in name_models_LR:
        n_par = fit_data['subject_level_parameter-estimates'][i].shape[0]
        n_params[i] = [n_par]
    n_subj = len([i for i in fit_data['subject_level_model_evidence']])
    
    res_lr = pd.DataFrame(index = ['lr','corr_lr','correction'])
    for model_LR in name_models_LR[1::]:
        model_ev_win = np.array(fit_data['group_level_model_evidence'][name_models_LR[0]])
        model_ev_win_Nparam = np.array(n_params[name_models_LR[0]])
        model_ev_cntrl = np.array(fit_data['group_level_model_evidence'][model_LR])
        model_ev_cntrl_Nparam = np.array(n_params[model_LR])
        lr_uncorr = model_ev_win-model_ev_cntrl
        exp_modelparam = np.exp(model_ev_win_Nparam-model_ev_cntrl_Nparam)
        correction = np.log(exp_modelparam**(np.log(n_subj)/2))
        corr_lr = lr_uncorr + correction
        res_list = [float(lr_uncorr),float(corr_lr),float(correction)]
        res_lr[str(name_models_LR[0]) + '_vs_' + str(model_LR)] = res_list
    used_data_lr = {'n_subj':n_subj,
                 'n_params':n_params,
                 'fit_data':fit_data}
    return {'corr_lr':res_lr ,
            'used_data':used_data_lr}


########## Experimental Below

def fit_data_CV_mult(VPN_dat):

    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT_CV,VIEW_DEPENDENT_CV
    from model_functions_BFGS import VIEW_DEPENDENTxCONTEXT_DEPENDENT_CV,VIEW_INDEPENDENT_CV
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENT_CV
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT_CV
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize
    np.random.seed(1993)

    unique_id, data, lbfgs_epsilon, verbose_tot = VPN_dat
    vpn = unique_id
    
    epsilon_param = lbfgs_epsilon
    x_0_bfgs = 0.5
    
    total_results = {}
    
    
    
    curr_data_vpn = data
    
    cv_score_view_ind_cont = []
    cv_score_view_dep = []
    cv_score_view_dep_cont=[]
    cv_score_view_ind = []
    cv_score_view_ind_dep = []
    cv_score_view_ind_dep_cont = []
    cv_score_rnd = []
    
    # for trial in trial
    ''' one trial needs to be subtracted from the data, 
    since we are deleting one trial'''
    
    trials_n = len(curr_data_vpn)-1

    #### debug
    test_L = []
    hold_L = []  
    
    for indx in range(trials_n):
        
        
        # holdout-data
        holdout_data = curr_data_vpn.copy()[indx,:]
        action = curr_data_vpn.copy()[indx,:][2]
        
        # training data
        train_data = np.delete(curr_data_vpn.copy(),indx,axis=0)
        #train_data = curr_data_vpn.copy()
        curr_index = indx
        
        # data import
        stim_IDs                = train_data[:,3].copy()   #stimulus IDs of winning model 
        new_ID                  = train_data[:,4].copy()   #trials where new ID is introduced 
        numb_prev_presentations = train_data[:,5].copy()   #number_of_prev_presentations
        stim_IDs_perspective    = train_data[:,0].copy()   #view dependent
        VPN_output              = train_data[:,2].copy()   #VPN answers
        verbose                 = False

        ##### Model Optim
        #data_raw.append()
        test_L.append(holdout_data)
        hold_L.append(train_data)
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
        #data_M1_debug[0] = curr_data_vpn[:,2].copy() 
        params_m_1 = res1[0]
        m_1 = VIEW_INDIPENDENTxCONTEXT(data_M1_debug, params_m_1)
        
        init_V_m_1 = 0
        init_C_m_1 = 0
        if indx == 0:
            init_V_m_1 += m_1[1]['init_val']['init_v']
        else:
            data_cv_score = m_1[1]['data_store_1'].loc[curr_index-1]
            init_V_m_1 = data_cv_score['history_V']
            init_C_m_1 = data_cv_score['history_C']
        
    
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

########################### rnd_choice ###################################################     
        '''for every answer predicted model prob of =.5'''
        ans_prob_rnd = np.log(.5)


##############################################################################        
        
        cv_score_view_ind_cont.append(cv_trial_indeXcontext)
        cv_score_view_dep.append(cv_trial_dep)
        cv_score_view_dep_cont.append(cv_trial_dep_cont)
        cv_score_view_ind.append(cv_trial_ind)
        cv_score_view_ind_dep.append(cv_trial_ind_dep)
        cv_score_view_ind_dep_cont.append(cv_trial_ind_dep_cont)
        cv_score_rnd.append(ans_prob_rnd)

    df_index = ['VIEW_INDIPENDENTxCONTEXT',
                'VIEW_DEPENDENT',
                'VIEW_DEPENDENTxCONTEXT_DEPENDENT',
                'VIEW_INDEPENDENT',
                'VIEW_INDEPENDENTxVIEW_DEPENDENT',
                'VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT',
                'RANDOM_CHOICE']
    
    df_data = [np.sum(cv_score_view_ind_cont),
               np.sum(cv_score_view_dep),
               np.sum(cv_score_view_dep_cont),
               np.sum(cv_score_view_ind),
               np.sum(cv_score_view_ind_dep),
               np.sum(cv_score_view_ind_dep_cont),
               np.sum(cv_score_rnd)]
    cv_trial = pd.DataFrame(data = df_data, index=df_index)
 
    total_results[vpn] = cv_trial
    total_results['suppl'] = (test_L,hold_L)
    total_results['error'] = (cv_score_view_ind_cont,df_data,m_1)
    total_results['VPN'] = curr_data_vpn
    return total_results

# debug = {'error?': (cv_score_view_ind_dep_cont,df_data,m_6),


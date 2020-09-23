# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:45:32 2020

@author: de_hauk
"""


def get_data(path):
    import pandas as pd
    ### Get data 
    data = pd.read_csv(path + r'/final_proc_dat_labjansen.csv')

    #### get unique IDS
    sample_answer_clms = [i for i in data.columns.values.tolist() if 'answer' in i]
    sample_perspective_clms = [i for i in data.columns.values.tolist() if 'perspective' in i]
    
    return {'data':data,
            'view_ind_data':sample_answer_clms,
            'view_dep_data':sample_perspective_clms}


def get_data_2(path_raw_data, ground_truth_file):
    import pandas as pd
    import glob
    import numpy as np
    from autoimpute.imputations import MultipleImputer
    data_path = path_raw_data
    #data_path = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
    all_files = glob.glob(data_path + "/*.csv")
    
    sample_fullinfo = pd.read_csv(ground_truth_file)
    SAMPLE_fullinfo = sample_fullinfo.drop(columns = ['Unnamed: 0']).copy()  
    
    #DATA_raw_DF = pd.DataFrame(SAMPLE_fullinfo).copy()
    DATA_raw_DF = pd.DataFrame()
    for data in all_files:
        unique_ID = data[-12] + '_' + data[-5] 
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
                
    
        DATA_raw_DF[unique_ID + 'perspective'] = perspective
        DATA_raw_DF[unique_ID + 'perf'] = answer_correct
        DATA_raw_DF[unique_ID + 'answer'] = answer_raw
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
    return DATA_imput

def data_old_sample(path):
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



def fit_data_noCV(data, lbfgs_epsilon, verbose_tot):
        
    from model_functions_BFGS import VIEW_INDIPENDENTxCONTEXT,VIEW_DEPENDENT,VIEW_DEPENDENTxCONTEXT_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENT, VIEW_INDEPENDENTxVIEW_DEPENDENT
    from model_functions_BFGS import VIEW_INDEPENDENTxVIEW_DEPENDENTxCONTEXT
    
    #folder_path_data = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\processed'
    
    
    
    import pandas as pd
    import numpy as np
    from functools import partial
    from scipy import optimize
    from scipy import special
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
        
        res_evidence[i] = [i[1] for i in [res1,res2,res3,res4,res5,res6]]
        
        trialwise_dat = {}
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
        
        #m_1[1]['data_store_1']
        trialwise_data[i] = wise_dat = m_1[1]['data_store_1']
    restotal = res_evidence.sum(axis=1)
    cntrl_log = special.logsumexp(-1*np.array(restotal[1::]))
    bf_log = (-1*np.array(restotal[0])) - cntrl_log
    if verbose_tot==True:
        return (restotal,data_verbose_debug)
    elif verbose_tot==False:
        return {'bf_log':bf_log,
                'subj_evidence':res_evidence,
                'total_evidence':restotal,
                'used_data': data,
                'parameters':parameter_est,
                'trialwise_dat':trialwise_data}
                

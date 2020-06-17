# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:01:21 2020

@author: hauke
"""


def find_sample(data, iterations, orig_dat):
    '''This Function takes the reversed input of probabilities and computes
    iteratively multiple lists that fit the original data of figure 1_C_right
    '''
    data=data
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    numb_IDs = 24
    num_iter = iterations
    prev_iter = []
    # iterate over groud data
    for it in tqdm(range(num_iter)):
        
        #progress bar per 10000 Iteration
        # if it % 10000 == 0:
        #     print(it)
        
        #iterate over each reconstructed prob & draw binomial where 1 == new face
        # new hypothetical data about when a new face is introduced
        Data_new_ID = np.random.binomial(1,data)

        #get amount of new presentation over the trials
        vals, cnts = np.unique(Data_new_ID,return_counts=True)
        
        # if counts of new presentations match the predefined amount of IDs (24) 
        # save data, else -> new it
        if int(cnts[1]) == int(numb_IDs):
            # check if first data in datastorage
            new_roll_avg = pd.Series(Data_new_ID).rolling(10, min_periods=10).mean().fillna(0)
            mse = np.sum((np.array(new_roll_avg)-np.array(orig_dat))**2)
            
            if not prev_iter:
                prev_iter.append((Data_new_ID,mse,new_roll_avg)) # store dat
                 
            # if not first entry in data, do same as above    
            elif prev_iter[-1][1] > mse:
                prev_iter.append((Data_new_ID,mse,new_roll_avg))
        else:
            None
            
    return prev_iter[-1]




def find_trial_list_189(generated_data, extr_data, iterations, start_point, steps):
    ''' This function uses the generated list of when a new facial stimuli is 
    introduced and tries to generate/ distribute the stimulus ids (1-24) over 
    the course of the experiment'''
    
    import numpy as np
    import pandas as pd
   
    ''' Input into function'''

    roll_avg_window = 5
    data = generated_data
    stim_info_15 = (15,12)
    stim_info_9 = (9,1)
    # Takes for stim_info_12,stim_info_9 tuple with each (stim amount, repetition amount)
    numb_stim = stim_info_15[0] + stim_info_9[0]

    ''' Protocol Methods Overall '''

    results_best = [] 
    
    for it in range(iterations):
        if it % 1000 == 0: #progress
            print(it)
    
        '''dict for trackkeeping @ each iteration '''
        curr_pres_num_dict = dict.fromkeys(np.arange(1,numb_stim+1,1,dtype=int),0 ) # current pres @ trial t
        
        ''' Protocol Methods per Iteration ''' 
        curr_dat = []
        curr_pres_numb = []
        counter = 1
        f_attemps = 0
        ''' Break current loop if not finished'''
        
        '''for each trial in current list'''
        for trial_type, trial_raw in zip(data, range(len(data))):
            trial = trial_raw+1
            '''optimization of start & step'''
            if (trial > start_point) and counter <= 99:
                counter +=steps
 
            if trial_type == 1: # if new face_ID 
                try:
                    ''' compile list with all eligible stimuli/ stimulus pool at trial t
                    with NO prior presentations'''
                    stim_pool1 = []
                    for key, value in curr_pres_num_dict.items():
                        if value == 0: # 0 prior presentations
                           stim_pool1.append(key)
                    
                    '''generate specific stim pools for 9*1 & 15*12 stimuli'''
                    stim_pool_15x12 = [i for i in stim_pool1 if i <= 15]
                    stim_pool_9x1 = [i for i in stim_pool1 if i >= 16]
                    '''random select stimulus from pool of new stimuli'''
                    # if no more 15*12 stimuli  -> take 9*1 stim 
                    if len(stim_pool_15x12) == 0:
                        
                        select_stim_1 = np.random.choice(stim_pool_9x1)
                        #update pres numb of stim
                        curr_pres_num_dict[select_stim_1] +=1
                        #append to representative list
                        curr_pres_numb.append(curr_pres_num_dict[select_stim_1])
                        curr_dat.append(select_stim_1)
        
                            
                    # while 15*12 stim left, prioritize 15*12 stim
                    elif len(stim_pool_15x12) >= 1:
                        
                        # counter to increse prob of 9*1 stim
                        choice_var = np.random.binomial(1,counter/100)
                        
                        # introduce 15x12 stim if available
                        if (choice_var == 0) and (len(stim_pool_15x12) > 0): 
                            select_stim_1 = np.random.choice(stim_pool_15x12)
                            #update pres numb of stim
                            curr_pres_num_dict[select_stim_1] +=1
                            #append to representative list
                            curr_pres_numb.append(curr_pres_num_dict[select_stim_1])
                            curr_dat.append(select_stim_1)
                           
                                
                         # introduce 9x1 stim if available and if trial >160
                        elif (trial > 160) and (choice_var == 1) and (len(stim_pool_9x1) > 0):
                            select_stim_1 = np.random.choice(stim_pool_9x1)
                            #update pres numb of stim
                            curr_pres_num_dict[select_stim_1] +=1
                            #append to representative list
                            curr_pres_numb.append(curr_pres_num_dict[select_stim_1])
                            curr_dat.append(select_stim_1)
                except:
                    None
                            
                        

            #runs if previously presented stim is to be presented
            elif trial_type == 0:
                try:
                    stim_pool0 = []
                    stim_pool_rep = []
                    for key, value in curr_pres_num_dict.items():
                        if (value >=1) and (value <12):
                            stim_pool0.append(key)
                            stim_pool_rep.append(value)
    
                    
    
                    idx = np.array([np.absolute(i-extr_data[trial_raw]) for i in stim_pool_rep]).argmin()
                    select_stim0 = np.array(stim_pool0)[idx]
    
                    #select_stim0 = np.random.choice(stim_pool0)
    
                    #update representation
                    curr_pres_num_dict[select_stim0] +=1
    
                    curr_pres_numb.append(curr_pres_num_dict[select_stim0])
                    curr_dat.append(select_stim0)
                except:
                    None
                   
        if len(curr_dat)==189 and len(curr_pres_numb) == 189:
            
            #rolling avg for m presentations t-5
            new_roll_avg = pd.Series(curr_pres_numb).rolling(roll_avg_window, min_periods=roll_avg_window).mean().fillna(0)
            mse = np.sum((np.array(new_roll_avg)-np.array(extr_data))**2)
            
            ## appends (stimulus_IDs, numb of presentations at each trial, stimulus presentation counts, MSE, new roll average comp from est data)
            if not results_best: # if first iteration/ list still empty
                results_best.append((curr_dat, curr_pres_numb,np.unique(curr_dat, return_counts = True),mse,new_roll_avg))
            else:
                # check for improvement over last iteration
                if mse <= results_best[-1][3]:
                    results_best.append((curr_dat, curr_pres_numb,np.unique(curr_dat, return_counts = True),mse,new_roll_avg,data))                        
        else:
            None

    return results_best

# from skopt import gp_minimize, utils, space
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# ### Get transcribed data from figure 1c right & left side
# data_raw_fig1_c_r = np.array(pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\Data_Gen\Fig_1_c_right\Fig_1_c_right.csv')['0'])
# data_raw_fig1_c_l = np.array(pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\Data_Gen\Fig_1_c_left\transcribed_raw_FIG_1_c_L.csv')['0'])
# window_fig1_c_r = 10
# window_fig1_c_l = 5
# blocks = 6
# trials = len(data_raw_fig1_c_r) 
# len_block = trials/blocks
# data_example = pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\data_lab_jansen\stimuli_list.csv')
# results_opt = find_trial_list_189(data_example['new_IDs'], data_raw_fig1_c_l, 20000, 142,7)

# fig,axs = plt.subplots(3,1, sharex = True, sharey=True)
# plt.xticks(np.arange(1,190,6))
# sns.lineplot(data=data_raw_fig1_c_l,ax = axs[0])
# sns.lineplot(data=np.array(results_opt[-1][4]),ax = axs[0])

# abc = results_opt[-1]


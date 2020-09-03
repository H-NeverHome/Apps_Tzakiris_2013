# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:56:44 2020

@author: de_hauk
"""


import pickle
import pandas as pd
import glob
import numpy as np
from autoimpute.imputations import MultipleImputer

data_path = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
all_files = glob.glob(data_path + "/*.csv")

sample_fullinfo = pd.read_csv(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')
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
            perspective.append(data_point[-1])
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
raw_dat_unproc = [pd.read_csv(i,header=None, sep='\t').drop(axis='index', labels = [0,1])[2] for i in all_files]

### determine place and amount of missing value
missing_dat_raw = pd.DataFrame(DATA_raw_DF.isnull().sum(), columns = ['dat'])

### filter out clmns with no missing dat
missing_dat_overview = missing_dat_raw.loc[missing_dat_raw['dat'] > 0].T

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

###############################################################################

def data_new():
    import pandas as pd
    import glob
    import numpy as np
    from autoimpute.imputations import MultipleImputer
    
    data_path = r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_raw\csv'
    all_files = glob.glob(data_path + "/*.csv")
    
    sample_fullinfo = pd.read_csv(r'C:\Users\de_hauk\PowerFolders\apps_tzakiris_rep\data\data_new_12_08_2020\data_list\stimuli_list.csv')
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

        
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:41:44 2021

@author: de_hauk
"""

##########
import pandas as pd
import numpy as np
from functools import partial
from scipy import optimize,special

import pandas as pd
np.random.seed(1993)


### Get data 
data_unform = data_processed
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
os.chdir(path_to_class)
#res_tot = [data_cv(i) for i in data_job[0:2]]  
if __name__ == '__main__':   
    from joblib import Parallel, delayed
    results123 = Parallel(n_jobs=8,verbose=50)(delayed(data_cv)(i) for i in data_job)    

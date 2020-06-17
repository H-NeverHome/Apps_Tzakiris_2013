# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:43:10 2020

@author: hauke
"""
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
data_raw_fig1_c_r = pd.Series(pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\Data_Gen\Fig_1_c_right\Fig_1_c_right.csv')['0'])
res = [(0.15,0.50)]
for it in tqdm(range(1000000)):
    init_val = np.random.uniform(0,1,189)
    draw_binom = pd.Series(np.random.binomial(1,data_raw_fig1_c_r))
    if np.unique(draw_binom,return_counts = True)[1][1] < 24:
        mov_avg = draw_binom.rolling(10,10).mean().fillna(0.5)
        mse = (np.absolute(mov_avg-data_raw_fig1_c_r)).sum()
        if mse < res[-1][1]:
            res.append((init_val,mse))
sns.lineplot(data=res[-1][0])
sns.lineplot(data=data_raw_fig1_c_r)     
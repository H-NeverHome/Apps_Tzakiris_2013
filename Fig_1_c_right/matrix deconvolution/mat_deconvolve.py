# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:24:03 2020

@author: hauke
"""

import seaborn as sns
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
#convolve(data, kernel)

### weekly oil prices
data_oil = pd.read_csv(r'https://datahub.io/core/oil-prices/r/brent-week.csv')['Price']

###### Compute roling avg with windowsize == 4 and 4 weeks lag
window = 10
lag = 0
### with pandas 
data_oil_roll_pd = data_oil.rolling(window,lag).mean()

### with matrix convolution
matrix_len = len(data_oil) # length of input
window_neg = -1*(window+1)
conv_mat = (np.tril(np.ones((matrix_len,matrix_len)),-1) - np.tril(np.ones((matrix_len,matrix_len)), window_neg)) /window
conv_mat_1 = np.matrix(conv_mat[window:,:])
conv_mat_inv = np.linalg.pinv(conv_mat_1)
 
data_oil_roll_mat_raw = list(pd.DataFrame(np.dot(data_oil,conv_mat_1.T)).T[0])
data_oil_roll_mat = np.array(data_oil_roll_mat_raw + [data_oil_roll_mat_raw[-1] for i in range(window)])

deconv = np.dot(data_oil_roll_mat,conv_mat_inv)
sns.lineplot(data=data_oil_roll_pd)
sns.lineplot(data=data_oil_roll_mat)



mse = ((data_oil_roll_pd - data_oil_roll_mat)**2).mean()

# #compute mask
# mask=np.ones((1,window))/window
# mask=mask[0,:]
# #Convolve the mask with the raw data
# convolved_data=np.convolve(data_oil,mask,'same')
# from skimage import color, data, restoration
# deconvolved_RL = restoration.richardson_lucy(data_oil_roll_mat, mask, iterations=30)
# # make pseudoinverse of convolution matrix
# mask_inv = np.linalg.pinv(mask)
'''
### Get Data from figure 1c right side
data_raw_fig1_c_r = np.array(pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\Data_Gen\Fig_1_c_right\Fig_1_c_right.csv')['0'])
data_raw_fig1_c_l = np.array(pd.read_csv(r'D:\Apps_Tzakiris_rep\A_T_Implementation\Data_Gen\Fig_1_c_left\transcribed_raw_FIG_1_c_L.csv')['0'])
window_fig1_c_r = 10
window_fig1_c_l = 5
###############################################################################
data_raw = data_raw_fig1_c_r
window = 10
#takes fractions as an input- return percentages
#source @ https://stats.stackexchange.com/questions/67907/extract-data-points-from-moving-average
#specifically this = https://stats.stackexchange.com/a/68002
import numpy as np
#data params
matrix_len = len(data_raw) # length of input
window_neg = -1*(window+1)
A = (np.tril(np.ones((matrix_len,matrix_len)),-1) - np.tril(np.ones((matrix_len,matrix_len)), window_neg)) /float(window)
A = A[window:,:]
pA = np.matrix(np.linalg.pinv(A)) #pseudo inverse

reconstr_avg_raw = np.dot(data_raw,pA)



reconstr_avg = np.append(reconstr_avg_raw,np.zeros(window))
# reconstruct original rolling avg datapoints from reconstructed datapoints
reconst_orig_datapoints = np.dot(pA,reconstr_avg_raw)



'''





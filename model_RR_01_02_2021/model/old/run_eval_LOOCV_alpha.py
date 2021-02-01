# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:21:43 2020

@author: de_hauk
"""







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
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:52:33 2021

@author: de_hauk
"""




for data in subjects:
    ## fit model with all trials, w/o imputed i.e. do not evaluate there
    
    for trial in trials:
        #get model prediction probability of current trial (MPP_yes)
        action_trial = rnd.binomial(1,pred_prob)
        #evaluate prediction 
        if action_trial == 1: #yes
            history_answer.append(np.log(MPP_yes))
        if action_trial == 0: #no
            history_answer.append(np.log(MPP_no))
    #sum evaluation over all trials 
    #Done?

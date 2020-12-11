# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:54:41 2020

@author: de_hauk
"""
import matlab.engine
import numpy as np
eng = matlab.engine.start_matlab()
#see https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/#between-conditions-rfx-bms
abcd= [[[10,11,12],[13,14,15],[16,17,18]],
                [[1,2,3,],[4,5,6],[7,8,9]],
                 [[1,2,3,],[4,5,6],[7,8,9]]]
aaaa = matlab.double(abcd)
func = eng.VBA_groupBMC_btwConds(aaaa)
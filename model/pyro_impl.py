# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:41:56 2020

@author: de_hauk
"""
import torch
import pyro

loc = 0.   # mean zero
scale = 1. # unit variance
normal =  pyro.distributions.Normal(loc, scale) # create a normal distribution object
x = normal.sample(sample_shape=torch.Size([100])) # draw a sample from N(0,1)
print("sample", x)
log_p = normal.log_prob(x).numpy()

#print("log prob", normal.log_prob(x)) # score the sample from N(0,1)
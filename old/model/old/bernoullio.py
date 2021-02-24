# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:52:33 2021

@author: de_hauk
"""
import numpy as np
import pymc3 as pm
import arviz as az

n_param = 100
data1 = np.random.binomial(n_param,.78)
data2 = np.random.binomial(n_param,.15)

# Fig 3.1
# if __name__ == '__main__':
#     with pm.Model() as binom_model:
#         theta = pm.Uniform('theta',lower=.0,upper=1.)

#         observed_y = pm.Bernoulli('obs',n=len(data1),p=theta,observed=data1)
#         idata = pm.sample(2000, tune=1500,cores=8,chains=8)
#     abc = az.plot_trace(idata)
#     aaaac = az.summary(idata)


# Fig 3.3
# if __name__ == '__main__':
#     with pm.Model() as binom_model2:
#         theta1 = pm.Beta('theta1', alpha=1, beta=1)
#         theta2 = pm.Beta('theta2', alpha=1, beta=1)
        
#         observed_1 = pm.Binomial('obs1',n=n_param,p=theta1,observed=data1)
#         observed_2 = pm.Binomial('obs2',n=n_param,p=theta2,observed=data2)
#         delta = pm.Deterministic('delta', theta1-theta2)
#         iidata = pm.sample(2000, tune=1500,cores=8,chains=8)
#     aaaacd = az.summary(iidata)


# Fig 3.5

if __name__ == '__main__':
    with pm.Model() as binom_model2:
        theta = pm.Beta('theta', alpha=1, beta=1)
        
        observed_1 = pm.Binomial('obs1',n=n_param,p=theta,observed=data1)
        observed_2 = pm.Binomial('obs2',n=n_param,p=theta,observed=data2)
        iidata = pm.sample(2000, tune=1500,cores=8,chains=8)
    aaaacd = az.summary(iidata)
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:00:22 2019

@author: u341138
"""

import sys
sys.path.extend(['C:\\Users\\u341138\\Dropbox\\Projects\\GPRDD'])

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import gaussian_processes_RDD as gp

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def class_label(x, k, theta):
    arg = k*(x+theta)
    return np.exp(arg) / (1+np.exp(arg)) < np.random.random(size=len(x))

n = 50
t = 0

xmin= -3.0
xmax = 3.0

b0 = 2*np.random.random()
b1 = 1+np.random.random()
x = np.sort(xmin + (xmax - xmin)*np.random.random(size=n))

xlabels = class_label(x, k=100.0, theta=t)
x1 = x[xlabels]
x2 = x[~xlabels]

noise_level = 0.9

mu1 = b0 + b1*x1
mu2 = b0 + b1*x2

y1 = np.random.normal(loc=mu1, scale=noise_level)
y2 = np.random.normal(loc=mu2, scale=noise_level)

y_raw = np.zeros((n))
y_raw[xlabels] = y1
y_raw[~xlabels] = y2

xpred = np.linspace(xmin, xmax, num=99)

#kfun = gp.gp_k_constant # this is essentially a t-test!
kfun = gp.gp_k_sq_exp


for gap in np.linspace(0.0, 10.0, num=4):      
    y = y_raw + gap*xlabels
    
    ## Do everything for continuous model
    
    # GP object
    gp_cont = gp.GaussianProcess(x, y, kfun=kfun)    


    # log evidence 
    log_E_cont = gp_cont.evidence(method='importance_sampling', priors={'noise': gp.prior_gamma, 'length_scale': gp.prior_invgamma}, 
                                 hyperparameters={'noise': {'scale': 9.0, 'shape': 0.5}, 
                                   'length_scale': {'scale': 1.0, 'shape': 1.0}}, nmcmc=1000) 
    
   
    # get & set optimal parameters according to grid search
    gp_cont_optpars = gp_cont.optimize_hyperparameters(kfun=kfun, method='grid_search')
    
    ## Do everything for discontinuous model
    
    # GP mixture object
    gp_disc = gp.GaussianProcessMixture(x=[x1, x2], y=[y[xlabels], y[~xlabels]], kfun=kfun)
        
    # log evidence    
    log_E_disc = gp_disc.evidence(method='importance_sampling', priors=[{'noise': gp.prior_gamma, 
                                                     'length_scale': gp.prior_invgamma}, 
                                                    {'noise': gp.prior_gamma, 
                                                     'length_scale': gp.prior_invgamma}], 
                                                   hyperparameters=[{'noise': {'scale': 1.0, 'shape': 1.0}, 
                                                     'length_scale': {'scale': 1.0, 'shape': 1.0}}, 
                                                    {'noise': {'scale': 9.0, 'shape': 0.5}, 
                                                     'length_scale': {'scale': 1.0, 'shape': 1.0}}],
                                                   nmcmc=1000) 
    
    # get & set optimal parameters according to grid search
    gp_disc_optpars = gp_disc.optimize_hyperparameters(kfun=kfun, method='grid_search')
    
    ## Do plotting for both models
       
    f, (ax1, ax2)= plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12, 6))
    
    gp_cont.plot_predictive(ax1, xpred=xpred)
    ax1.set_title('Continuous model, (l,s) = ({:0.1f},{:0.1f})'.format(gp_cont_optpars['length_scale'], gp_cont_optpars['noise']))
    gp_disc.plot_predictive(ax2, xpred=xpred, t=t)
    ax2.set_title('Discontinuous model, (l,s)_1 = ({:0.1f},{:0.1f}) & (l,s)_2 = ({:0.1f},{:0.1f}) '.format(gp_disc_optpars[0]['length_scale'], gp_disc_optpars[0]['noise'], gp_disc_optpars[1]['length_scale'], gp_disc_optpars[1]['noise']))
    
    for ax in (ax1, ax2):
        ax.set_xlabel('predictor')
        ax.set_ylabel('response')
        ax.set_xlim([xmin, xmax])
        ax.scatter(x[~xlabels], y[~xlabels], marker='x', c='k')
        ax.scatter(x[xlabels], y[xlabels], marker='o', c='k')
        ax.axvline(x=t, color='k', linestyle='--')
    f.suptitle('Log Bayes factor in favor of discontinuity = {:0.2f}'.format(log_E_disc - log_E_cont), fontsize=16)
    plt.show()
    f.savefig('GPRDD_{:0.0f}.svg'.format(gap), bbox_inches='tight')    
        
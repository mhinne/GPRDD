# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:14:32 2019

@author: u341138
"""

import numpy as np

# kernel functions    
def gp_k_sq_exp(x1, x2, gppars):
    l = gppars['length_scale']
    return np.exp(- (x1 - x2)**2 / (2*l**2))

def gp_k_matern_52(x1, x2, gppars):
    r = x1 - x2
    l = gppars['length_scale']
    return (1+ np.sqrt(5)*r / l + 5*r**2 / (2*l**2)) * np.exp(-np.sqrt(5)*r/l)

class GaussianProcess:
    
    # Gaussian process object
    def __init__(self, x, y, kfun=None, pars=None):
        self.x = x
        self.y = y
        self.pars = pars
        n = len(x)
        self.n = n
        self.kfun = kfun
        self.__set_kernel(kfun, pars)
        self.f_post = None
        self.K_post = None
        
    def set_params(self, gppars):
        self.pars = gppars
        
    def posterior(self, cov=True):
        noise = self.pars['noise']
        n = len(self.y)
        K_noise = self.K + noise*np.eye(n)
        f_post = self.K @ np.linalg.solve( K_noise, self.y )
        self.f_post = f_post
    
        if not cov:            
            return f_post
        else:
            K_post = self.K - self.K.T @ np.linalg.solve(K_noise, self.K)
            self.K_post = K_post
            return f_post, K_post
        
    def __set_kernel(self, kfun, gppars=None):
        if gppars is None:
            gppars = self.pars
        else:
            self.pars = gppars
        n = self.n
        x = self.x
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = kfun(x[i], x[j], gppars)
        self.K = K
    
    def predictive(self, xpred):
        x = self.x
        n = self.n
        kfun = self.kfun
        xpred = np.atleast_1d(xpred)
        npred = len(xpred)
        Sigma = self.pars['noise'] * np.eye(n)
    
        K_cross = np.zeros((n, npred))
        for i in range(n):
            for j in range(npred):
                K_cross[i,j] = kfun(x[i], xpred[j], self.pars)
    
        K_pred = np.zeros((npred, npred))
        for i in range(npred):
            for j in range(npred):
                K_pred[i,j] = kfun(xpred[i], xpred[j], self.pars)
    
        # Rasmussen & Williams 2004, p 19
        L = np.linalg.cholesky( self.K + Sigma )
        alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, self.y)), (n, 1))
    
        # predictive mean
        f_pred = K_cross.T @ alpha
    
        # predictive variance
        v = np.linalg.solve(L, K_cross)
        var_pred = K_pred - v.T @ v
    
        return f_pred, var_pred
        
    def log_marginal_likelihood(self):
        x = self.x
        y = self.y
        n = len(x)
        Sigma = self.pars['noise'] * np.eye(n)
        L = np.linalg.cholesky(self.K + Sigma)
        alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, y)), (n, 1))
        log_evidence = -1/2 * y.T @ alpha - np.sum(np.diag(L)) - n/2*np.log(2*np.pi)
        
        return log_evidence
    
    def integrate_hyperparameters(priors, hyperparameters, nmcmc=1000):
        # TODO: make generic
        return 0.0
#        shape_ls = 1.0
#        scale_ls = 1.0
#        shape_nl = 0.5
#        scale_nl = 1.0
#        
#        log_marg_lik = np.zeros((nmcmc))
#        
#        length_scales = np.zeros((nmcmc))
#        noise_levels = np.zeros((nmcmc))
#        
#        for i in range(nmcmc):
#            length_scale = 1 / np.random.gamma(shape=shape_ls, scale=scale_ls, size=1)
#            noise_level = np.random.gamma(shape=shape_nl, scale=scale_nl)
#            gppars = {'noise': noise_level, 'length_scale': length_scale}
#            K = gp_kernel(x, kfun, gppars)
#            log_marg_lik[i] = log_marginal_likelihood(x, y, K, gppars)
#            length_scales[i] = length_scale
#            noise_levels[i] = noise_level
#        
#        marg_lik = np.exp(log_marg_lik)    
#        return marg_lik, length_scales, noise_levels  
    
    def plot_posterior(self, axis, col=(0.9, 0.1, 0.1)):
        
        if self.f_post is None:
            self.posterior(cov=True)
        
        r, g, b = col
        x = self.x
        f = self.f_post
        var = self.K_post
        if var is not None:
            ub = f + np.sqrt(np.diag(var))
            lb = f - np.sqrt(np.diag(var))
            axis.fill_between(x, lb, ub, color=(r, g, b, 0.3), linewidth=0)
        axis.plot(x, f, color=(r, g, b, 0.9))
    
    
    
        
        
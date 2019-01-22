# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:14:32 2019

@author: u341138
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# kernel functions    
def gp_k_sq_exp(x1, x2, gppars):
    l = gppars['length_scale']
    return np.exp(- (x1 - x2)**2 / (2*l**2))

def gp_k_matern_52(x1, x2, gppars):
    r = x1 - x2
    l = gppars['length_scale']
    return (1+ np.sqrt(5)*r / l + 5*r**2 / (2*l**2)) * np.exp(-np.sqrt(5)*r/l)

def gp_k_linear(x1, x2, gppars):
    offset = gppars['offset']
    return offset + x1 - x2

def gp_k_constant(x1, x2, gppars):
    c = gppars['constant']
    return c

# hyperparameter prior distributions
    
def prior_gamma(params):
    sample = np.random.gamma(shape=params['shape'], scale=params['scale'], size=1)
    density = stats.gamma.pdf(sample, params['shape'], 0.0, params['scale'])
    return sample, density

def prior_invgamma(params):
    sample = 1 / np.random.gamma(shape=params['shape'], scale=params['scale'], size=1)
    density = stats.invgamma.pdf(sample, params['shape'], 0.0, params['scale'])
    return sample, density

def prior_normal(params):
    sample = np.random.normal(loc=params['loc'], scale=params['scale'], size=1)
    density = stats.norm.pdf(sample, params['loc'], scale=params['scale'])
    return sample, density

class GaussianProcess:
    
    # Gaussian process object
    def __init__(self, x, y, kfun=None, pars=None):
        self.x = x
        self.y = y
        self.pars = pars
        n = len(x)
        self.n = n
        self.kfun = kfun
        if pars is not None:
            self.K = self.__construct_kernel(kfun, pars)
        else:
            self.K = None
        self.f_post = None
        self.K_post = None
        self.f_pred = None
        self.var_pred = None
        self.f_margin_pred = None
        self.log_E = None
        
        
    def __construct_kernel(self, kfun, gppars=None):
        if gppars is None:
            gppars = self.pars
        n = self.n
        x = self.x
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = kfun(x[i], x[j], gppars)
        return K
    
    
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
    
    
    def predictive(self, xpred, KK=None):
        x = self.x
        n = self.n
        kfun = self.kfun
        xpred = np.atleast_1d(xpred)
        npred = len(xpred)
        Sigma = self.pars['noise'] * np.eye(n)
        
        if KK is None:
            if self.K is None:
                self.K = self.__construct_kernel(kfun, self.pars)
            KK = self.K
    
        K_cross = np.zeros((n, npred))
        for i in range(n):
            for j in range(npred):
                K_cross[i,j] = kfun(x[i], xpred[j], self.pars)
    
        K_pred = np.zeros((npred, npred))
        for i in range(npred):
            for j in range(npred):
                K_pred[i,j] = kfun(xpred[i], xpred[j], self.pars)
    
        # Rasmussen & Williams 2004, p 19
        L = np.linalg.cholesky( KK + Sigma )
        alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, self.y)), (n, 1))
    
        # predictive mean
        f_pred = K_cross.T @ alpha
    
        # predictive variance
        v = np.linalg.solve(L, K_cross)
        var_pred = K_pred - v.T @ v
        
        self.f_pred = f_pred
        self.var_pred = var_pred
    
        return f_pred, var_pred
        
    
    def log_marginal_likelihood(self, K=None, gppars=None):
        # See Rasmussen & Williams, p. 19
        if K is None:
            K = self.K
        if gppars is None:
            gppars = self.pars
        x = self.x
        y = self.y
        n = len(x)
        Sigma = gppars['noise'] * np.eye(n)
        L = np.linalg.cholesky(K + Sigma)
        alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, y)), (n, 1))
        log_evidence = -1/2 * y.T @ alpha - np.sum(np.diag(L)) - n/2*np.log(2*np.pi)
        
        return log_evidence
    
    
    def eval_params(self, gppars):
        KK = self.__construct_kernel(self.kfun, gppars)
        return self.log_marginal_likelihood(KK, gppars)
    
    
    def evidence(self, **kwargs):
        print('Approximating marginal likelihood')
        method = kwargs['method']
        if method is 'importance_sampling':
            return self.integrate_hyperparameters(kwargs['priors'], kwargs['hyperparameters'], kwargs['nmcmc'])
        else:
            print('Sorry, {} is not yet implemented.'.format(method))
            return np.NAN
        
    
    def integrate_hyperparameters(self, priors, hyperparameters, nmcmc=1000):        
        self.param_samples = {}        
        for param in priors.keys():
            self.param_samples[param] = np.zeros((nmcmc))
        
        # Unnormalized log (marginal over f) likelihood, as samples are drawn from prior p(theta) 
        log_marg_liks = np.zeros((nmcmc))
        self.stored_kernels = []       
        
        for i in range(nmcmc):
            gppars = {}
            for param in priors.keys():
                sample, _ = (priors[param])(hyperparameters[param])
                gppars[param] = sample
                self.param_samples[param][i] = sample
            KK = self.__construct_kernel(self.kfun, gppars)
            log_marg_liks[i] = self.log_marginal_likelihood(KK, gppars)
            self.stored_kernels += [KK]

        # for numerical stability, see e.g. https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        a = np.max(log_marg_liks)          
        self.log_E = a + np.log(np.mean(np.exp(log_marg_liks - a)))        
        self.importance_weights = log_marg_liks
        
        return self.log_E
    
    
    def optimize_hyperparameters(self, kfun, method='grid_search', grid_granularity=30, plot=False):
        print('Optimizing hyperparameters')
        if method is 'grid_search':
            if kfun == gp_k_sq_exp:                
                ls_range = np.logspace(-4, 1, num=grid_granularity)
                nl_range = np.logspace(-1, 1, num=grid_granularity)
                
                m = len(ls_range)
                n = len(nl_range)
                lik = np.ndarray((m, n))
                
                for i, length_scale in enumerate(ls_range):
                    for j, noise_level in enumerate(nl_range):
                        gppars = {'length_scale': length_scale, 'noise': noise_level}
                        KK = self.__construct_kernel(kfun, gppars)
                        lik[i,j] = self.log_marginal_likelihood(KK, gppars)
                
                i,j = np.unravel_index(np.argmax(lik.flatten()), dims = np.shape(lik))    
                optpars = {'length_scale': ls_range[i], 'noise': nl_range[j]}
                
                if plot:
                    gridimg = plt.imshow(lik, origin='upper')
                    plt.xticks(range(m), ls_range, rotation=90)
                    plt.yticks(range(n), nl_range)
                    gridimg.set_cmap('hot')
                    plt.colorbar()
                    plt.scatter(j, i, marker='s', facecolors='none', edgecolors='k')
                    plt.xlabel('length scale')
                    plt.ylabel('noise level')
                    plt.title('Likelihood grid')
                
                self.pars = optpars
                return optpars
            else:
                print('Optimization for this kernel is not yet implemented')
                return None
        else:
            print('This optimization method is not yet implemented')
            return None
    
    
    def get_expected_params(self):
        w = self.importance_weights - self.log_E
        gppars_expected = {}
        for param in self.param_samples.keys():
            for i in range(len(self.importance_weights)):
                gppars_expected[param] = (self.param_samples[param])[i]*np.exp(w[i])
        return gppars_expected
    
    
    def marginalized_predictive(self, xpred):
        assert self.log_E is not None, 'Marginal likelihood not yet computed; call gp.evidence(...) first.'        

        nmcmc = len(self.importance_weights)        
        w = np.exp(self.importance_weights - self.log_E)
        n = len(xpred)
        
        f_margin_pred = np.ndarray((n, 1), dtype=np.float64)
        var_margin_pred = np.ndarray((n,n), dtype=np.float64)
        
        for i in range(nmcmc):
            KK = self.stored_kernels[i]
            f_i, var_i = self.predictive(xpred, KK)
            f_margin_pred += w[i]/nmcmc *  f_i
            var_margin_pred += w[i]/nmcmc * var_i
            
        self.f_margin_pred = f_margin_pred
        self.var_margin_pred = var_margin_pred
        return f_margin_pred, var_margin_pred    
    
    
    def plot_posterior(self, axis, col=(0.9, 0.1, 0.1)):        
        if self.f_post is None:
            self.posterior(cov=True)
        
        r, g, b = col
        x = self.x
        f = self.f_post
        var = self.K_post
        if var is not None:
            ub = f + np.sqrt(np.abs(np.diag(var)))
            lb = f - np.sqrt(np.abs(np.diag(var)))
            axis.fill_between(x, lb, ub, color=(r, g, b, 0.3), linewidth=0)
        axis.plot(x, f, color=(r, g, b, 0.9))
        
    
    def plot_predictive(self, axis, xpred, col=(0.9, 0.1, 0.1)):        
        #if self.f_pred is None:
        self.predictive(xpred)        
        r, g, b = col
        npred = len(xpred)
        xpred = np.reshape(xpred, newshape=(npred,1))
        f_pred = np.reshape(self.f_pred, newshape=(npred, 1))
        var_pred = self.var_pred
        if var_pred is not None:
            ub = f_pred + np.reshape(np.sqrt(np.abs(np.diag(var_pred))), newshape=(npred, 1))
            lb = f_pred - np.reshape(np.sqrt(np.abs(np.diag(var_pred))), newshape=(npred, 1))
            axis.fill_between(xpred.flatten(), lb.flatten(), ub.flatten(), color=(r, g, b, 0.2), linewidth=0)
        axis.plot(xpred.flatten(), f_pred.flatten(), color=(r, g, b, 0.9))   
    
    def plot_marginalized_predictive(self, axis, xpred, col=(0.9, 0.1, 0.1)):
        # TODO: plot prediction over integrated hyperparameters, i.e.
        # p(f* | X, y) = \int p(f* | X, y, theta) p(theta | X, y) d theta
        # -> we can do this now we have the evidence terms
        # We have computed/available log p(y | X, theta) = log \int p(y | f, X, theta) p(f | X, theta) p(theta) d theta
        # and we have p(theta | X) = p(theta)
        # so we can use the prior samples and give them weights log_marg_lik[i] / evidence to reflect posterior
        # nb: the scaling by the evidence is done for all samples so can be ignored?
        
        # Can we just compute the predictive weighted by normalized likelihoods?
        
        if self.f_margin_pred is None:
            self.marginalized_predictive(xpred)
        assert self.log_E is not None, 'Marginal likelihood not yet computed; call gp.evidence(...) first.'
        
        r, g, b = col
        npred = len(xpred)
        xpred = np.reshape(xpred, newshape=(npred,1))
        f_margin_pred = np.reshape(self.f_margin_pred, newshape=(npred, 1))
        var_margin_pred = self.var_margin_pred
        if var_margin_pred is not None:
            ub = f_margin_pred + np.reshape(np.sqrt(np.diag(var_margin_pred)), newshape=(npred, 1))
            lb = f_margin_pred - np.reshape(np.sqrt(np.diag(var_margin_pred)), newshape=(npred, 1))
            axis.fill_between(xpred.flatten(), lb.flatten(), ub.flatten(), color=(r, g, b, 0.2), linewidth=0)
        axis.plot(xpred.flatten(), f_margin_pred.flatten(), color=(r, g, b, 0.9))   
        
            
    
class GaussianProcessMixture:
    def __init__(self, x, y, kfun, pars=None):
        self.ncomponents = len(x)        
        gps = [None] * self.ncomponents        
        for i in range(self.ncomponents): 
            if pars is None:
                gps[i] = GaussianProcess(x[i], y[i], kfun)
            else:
                gps[i] = GaussianProcess(x[i], y[i], kfun, pars[i])
        self.components = gps
        
    def posterior(self, cov=True):
        for i in range(self.ncomponents):
            self.components[i].posterior()
            
    def predictive(self, xpred):
        for i in range(self.ncomponents):
            self.components[i].predictive(xpred)
            
    def marginalized_predictive(self, xpred):
        for i in range(self.ncomponents):
            self.components[i].marginalized_predictive(xpred)
    
    def plot_posterior(self, axis, col=(0.1, 0.9, 0.1)):
        for i in range(self.ncomponents):
            self.components[i].plot_posterior(axis, col)
    
    def plot_predictive(self, axis, xpred, t, col=(0.1, 0.9, 0.1)):
        # the selection only works for 2 components
        for i in range(self.ncomponents):
            self.components[i].plot_predictive(axis, xpred[((1-i)*(xpred<=t) + i*(xpred>=t)).astype(bool)], col)
            
    def plot_marginalized_predictive(self, axis, xpred, t, col=(0.1, 0.9, 0.1)):
        # the selection only works for 2 components
        for i in range(self.ncomponents):
            self.components[i].plot_marginalized_predictive(axis, xpred[((1-i)*(xpred<=t) + i*(xpred>=t)).astype(bool)], col)
            
    def optimize_hyperparameters(self, kfun, method='grid_search', grid_granularity=30, plot=False):
        optpars = []
        for i in range(self.ncomponents):
            optpars += [self.components[i].optimize_hyperparameters(kfun, method, grid_granularity, plot)]
        return optpars
        
    def evidence(self, **kwargs):
        method = kwargs['method']
        if method is 'importance_sampling':
            log_marg_lik = 0
            priors = kwargs['priors']
            hyperparameters = kwargs['hyperparameters']
            nmcmc = kwargs['nmcmc']
            for i in range(self.ncomponents):
                log_marg_lik += self.components[i].integrate_hyperparameters(priors[i], hyperparameters[i], nmcmc)
            return log_marg_lik
        else:
            print('Sorry, {} is not yet implemented.'.format(method))
            return np.NAN
        
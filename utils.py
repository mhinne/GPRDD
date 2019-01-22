
# Python import nightmare
import sys
sys.path.extend(['C:\\Users\\u341138\\Dropbox\\Projects\\GPRDD'])

# Note: script uses python > 3.5 syntax for matrix operations

import numpy as np
#import matplotlib.pyplot as plt

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def simulate_data(n=100, t=0, gap=None, noise=1.0):
    b0 = 2*np.random.random()
    b1 = np.random.random()
    if gap is None:
        gap = np.random.random()
    x = np.sort(-3 + 6*np.random.random(size=n))
    mu = b0 + b1*x + gap*(x>t)
    y = np.random.normal(loc = mu, scale = noise, size=n)
    return x, y


def gp_k_sq_exp(x1, x2, gppars):
    l = gppars['length_scale']
    return np.exp(- (x1 - x2)**2 / (2*l**2))


def gp_k_matern_52(x1, x2, gppars):
    r = x1 - x2
    l = gppars['length_scale']
    return (1+ np.sqrt(5)*r / l + 5*r**2 / (2*l**2)) * np.exp(-np.sqrt(5)*r/l)


def gp_kernel(x, kfun, gppars):
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kfun(x[i], x[j], gppars)
    return K


def gp_posterior(y, K, gppars, cov=True):
    noise = gppars['noise']
    n = len(y)
    K_noise = K + noise*np.eye(n)
    f_post = K @ np.linalg.solve( K_noise, y )
    K_post = K - K.T @ np.linalg.solve(K_noise, K)

    if not cov:
        return f_post
    else:
        return f_post, K_post


def gp_predictive(x, y, xpred, K, kfun, gppars):
    n = len(x)
    xpred = np.atleast_1d(xpred)
    npred = len(xpred)
    Sigma = gppars['noise'] * np.eye(n)

    K_cross = np.zeros((n, npred))
    for i in range(n):
        for j in range(npred):
            K_cross[i,j] = kfun(x[i], xpred[j], gppars)

    K_pred = np.zeros((npred, npred))
    for i in range(npred):
        for j in range(npred):
            K_pred[i,j] = kfun(xpred[i], xpred[j], gppars)

    # Rasmussen & Williams 2004, p 19
    L = np.linalg.cholesky( K + Sigma )
    alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, y)), (n, 1))

    # predictive mean
    f_pred = K_cross.T @ alpha

    # predictive variance
    v = np.linalg.solve(L, K_cross)
    var_pred = K_pred - v.T @ v

    return f_pred, var_pred


def gp_plot_fit(axis, x, f, var=None, col=(0.9, 0.1, 0.1)):
    r, g, b = col
    if var is not None:
        ub = f + np.sqrt(np.diag(var))
        lb = f - np.sqrt(np.diag(var))
        axis.fill_between(x, lb, ub, color=(r, g, b, 0.3), linewidth=0)
    axis.plot(x, f, color=(r, g, b, 0.9))


def log_marginal_likelihood(x, y, K, gppars):
    n = len(x)
    Sigma = gppars['noise'] * np.eye(n)
    L = np.linalg.cholesky(K + Sigma)
    alpha = np.reshape(np.linalg.solve(L.T, np.linalg.solve(L, y)), (n, 1))
    evidence = -1/2 * y.T @ alpha - np.sum(np.diag(L)) - n/2*np.log(2*np.pi)
    return evidence


def integrate_hyperparameters(x, y, kfun, nmcmc=1000):    
    shape_ls = 1.0
    scale_ls = 1.0
    shape_nl = 0.5
    scale_nl = 1.0
    
    log_marg_lik = np.zeros((nmcmc))
    
    length_scales = np.zeros((nmcmc))
    noise_levels = np.zeros((nmcmc))
    
    for i in range(nmcmc):
        length_scale = 1 / np.random.gamma(shape=shape_ls, scale=scale_ls, size=1)
        noise_level = np.random.gamma(shape=shape_nl, scale=scale_nl)
        gppars = {'noise': noise_level, 'length_scale': length_scale}
        K = gp_kernel(x, kfun, gppars)
        log_marg_lik[i] = log_marginal_likelihood(x, y, K, gppars)
        length_scales[i] = length_scale
        noise_levels[i] = noise_level
    
    marg_lik = np.exp(log_marg_lik)    
    return marg_lik, length_scales, noise_levels    


def optimize_length_scale(x, y, kfun):
    noise = 0.2
    nmcmc = 1000
    
    length_scales = np.zeros((nmcmc))
    log_marg_lik = np.zeros((nmcmc))
    
    # this harmonic mean approach is probably not correct
    
    for i, length_scale in enumerate(np.linspace(0.01, 10, num=nmcmc)):
        #length_scale = 1 / np.random.gamma(shape=shape, scale=scale, size=1)
        gppars = {'noise': noise, 'length_scale': length_scale}
        K = gp_kernel(x, kfun, gppars)
        log_marg_lik[i] = log_marginal_likelihood(x, y, K, gppars)
        length_scales[i] = length_scale
    ix = np.argmax(log_marg_lik)
    return length_scales[ix]


def optimize_length_scale_LOO(x, y, kfun):
    # TODO: implement
    return 0.0
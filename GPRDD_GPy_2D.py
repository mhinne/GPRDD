# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:48:32 2019

@author: u341138
"""

import sys
sys.path.extend(['C:\\Users\\Max\\Dropbox\\Projects\\GPRDD'])
sys.path.extend(['C:\\Users\\u341138\\Dropbox\\Projects\\GPRDD'])
sys.path.extend(['C:\\Users\\Max Hinne\\Dropbox\\Projects\\GPRDD'])

import GPy
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import GPRDDAnalysis

print("GPy version:      {}".format(GPy.__version__))
print("GPRDD version:    {}".format(GPRDDAnalysis.__version__))
   

#
def generate_data(n, labelFunc, ndim=2, snr=1.0):   
    
    xmin    = -1.5
    xmax    = 1.5
    x      = np.random.uniform(xmin, xmax, size=(n, ndim))  
        
    noise   = np.random.normal(size=n)    
    gap     = snr
    
    b = np.array([5, 1.2, 1.8])
    b = np.expand_dims(b, axis=1)
    y = np.squeeze(b[0] + b[1:,].T @ x.T + noise)
    
    labels                 = labelFunc(x)
    y[labels] += gap
    y = np.expand_dims(y, axis=1)
    
    return x, y, labels
#

# TODO: generic usage requires that we apply this function to the test data as well -> can define any function we want here
labelFunc = lambda x: np.logical_and(x[:,0] < 0, x[:,1] > 0)
#labelFunc = lambda x: (x[:,0] + 2*x[:,1]) <= 0
#labelFunc = lambda x: (x[:,0]+1.0)**2 + + (x[:,1]-1.0)**2 <= 1

kerneltype = 'Matern32'

if kerneltype == 'Matern32':
    kernel = GPy.kern.Matern32(2) + GPy.kern.White(2)
elif kerneltype in ('Linear', 'Plane'):
    kernel = GPy.kern.Linear(2) + GPy.kern.Bias(2) + GPy.kern.White(2)
elif kerneltype in ('ttest', 'Constant'):
    kernel = GPy.kern.Bias(2) + GPy.kern.White(2)
elif kerneltype == 'RBF':
    kernel = GPy.kern.RBF(2) + GPy.kern.White(2)


snr = 3.0
n = 100

x, y, _ = generate_data(n, labelFunc, snr=snr)      

colmap = {0: 'blue', 1: 'red'}
cols = [colmap[i] for i in labels]

n_test = 1000
xx = np.linspace(-1.5, 1.5, num=np.sqrt(n_test))
X, Y = np.meshgrid(xx, xx)
x_test = np.array([X.flatten(), Y.flatten()]).T


gprdd = GPRDDAnalysis.GPRDDAnalysis(x, y, kernel, labelFunc)
gprdd.train()
gprdd.plot(x_test)


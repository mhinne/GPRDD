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
import matplotlib.pyplot as plt

import GPRDDAnalysis

print("GPy version:      {}".format(GPy.__version__))
print("GPRDD version:    {}".format(GPRDDAnalysis.__version__))
   

#
def generate_data_1D_disc(n, labelFunc, gap=1.0):    
    xmin    = -1.5
    xmax    = 1.5
    x       = np.linspace(xmin, xmax, num=n)    
    noise   = np.random.normal(size=n)   
    
    b0, b1, b2 = (5, 1.2, 0.0)
    y = b0 + b1*x + b2*np.sin(x) + noise
    
    labels                      = labelFunc(x)
    y[labels] += gap
    
    return x, y, labels
#
def generate_data_1D_diffslope(n, labelFunc, slopeRatio=2.0):    
    xmin    = -1.5
    xmax    = 1.5
    x       = np.linspace(xmin, xmax, num=n)    
    noise   = np.random.normal(size=n)   
    
    b0, b1= (5, 1.2)
    y = b0 + b1*x[labelFunc(x)] + b1*slopeRatio*x[np.logical_not(labelFunc(x))] + noise
    
    labels                      = labelFunc(x)
#    y[labels] += gap
    
    return x, y, labels
#


x_test = np.linspace(-1.5, 1.5, num=100)  


kerneltype = 'Matern32'

if kerneltype == 'Matern32':
    kernel = GPy.kern.Matern32(1) + GPy.kern.White(1)
elif kerneltype == 'Linear':
    kernel = GPy.kern.Linear(1) + GPy.kern.Bias(1) + GPy.kern.White(1)
elif kerneltype in ('ttest', 'Constant'):
    kernel = GPy.kern.Bias(1) + GPy.kern.White(1)
elif kerneltype == 'RBF':
    kernel = GPy.kern.RBF(1) + GPy.kern.White(1)


labelFunc = lambda x: x > 0

x, y, _ = generate_data_1D_disc(25, labelFunc, gap=1.0)      

gprdd = GPRDDAnalysis.GPRDDAnalysis(x, y, kernel, labelFunc)
gprdd.train()
gprdd.plot(x_test)


x, y, _ = generate_data_1D_diffslope(25, labelFunc, slopeRatio=2.0)      

gprdd = GPRDDAnalysis.GPRDDAnalysis(x, y, kernel, labelFunc)
gprdd.train()
gprdd.plot(x_test)




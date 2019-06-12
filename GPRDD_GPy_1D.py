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
def generate_data(n, labelFunc, snr=1.0):    
    xmin    = -1.5
    xmax    = 1.5
    x       = np.linspace(xmin, xmax, num=n)    
    noise   = np.random.normal(size=n)    
    gap     = snr
    
    b0, b1, b2 = (5, 1.2, 0.0)
    y = b0 + b1*x + b2*np.sin(x) + noise
    
    labels                      = labelFunc(x)
    y[labels] += gap
    
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


snr = 3.0
labelFunc = lambda x: x > 0

x, y, _ = generate_data(25, labelFunc, snr=snr)      




gprdd = GPRDDAnalysis.GPRDDAnalysis(x, y, kernel, labelFunc)
gprdd.train()
gprdd.plot(x_test)



#gprdd_analysis = GPRDD2D.GPRDDAnalysis(x, y, kernel, threshold=0.0)
#gprdd_analysis.run()
#
#                   
#gprdd_analysis.visualize(x_test, true_gap=snr)

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:04:20 2019

@author: u341138
"""

import sys
sys.path.extend(['C:\\Users\\Max\\Dropbox\\Projects\\GPRDD'])
sys.path.extend(['C:\\Users\\u341138\\Dropbox\\Projects\\GPRDD'])
sys.path.extend(['C:\\Users\\Max Hinne\\Dropbox\\Projects\\GPRDD'])

import GPy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv

import importlib
import GPRDDAnalysis
importlib.reload(GPRDDAnalysis)

print("GPy version:      {}".format(GPy.__version__))
print("GPRDD version:    {}".format(GPRDDAnalysis.__version__))

   

datafile = 'datasets\\sicilysmokingban\\sicily.csv'
data = dict()

with open(datafile, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for k, v in dict(row).items():
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

data['year']    = np.array([int(y) for y in data['year']])
data['month']   = np.array([int(m) for m in data['month']])
data['aces']    = np.array([int(a) for a in data['aces']]) # acute coronary events
data['time']    = np.array([int(t) for t in data['time']]) # predictor
data['smokban'] = np.array([s=='1' for s in data['smokban']])
data['pop']     = np.array([float(t) for t in data['pop']])
data['stdpop']  = np.array([float(st) for st in data['stdpop']]) # age-standardized population numbers


zscorex = lambda x: (x - np.mean(data['time'])) / np.std(data['time'])
zscorey = lambda x: (x - np.mean(aces_per_agestd_pop)) / np.std(aces_per_agestd_pop)

n = len(data['time'])

b = data['time'][data['smokban']][0] - 1 # the ban instantiated *before* this

bz = zscorex(b)

labelFunc = lambda x: x > bz

aces_per_agestd_pop = data['aces'] / data['stdpop']*10**5

x = zscorex(data['time'])
y = stats.zscore(aces_per_agestd_pop) # ACE / age-standardized population


kernel = GPRDDAnalysis.get_kernel(kerneltype='Periodic', D=1) # black magic?
gprdd = GPRDDAnalysis.GPRDDAnalysis(x, y, kernel, labelFunc)
gprdd.train(num_restarts=20) # needs higher number for publication results

x_test = np.linspace(x[0], x[-1], num=100)  
fig, axes = gprdd.plot(x_test, plotEffectSize=True, b=bz)

for ax in axes[0], axes[1]:
    ax.annotate('January 2005', xy=(bz, 2), xytext=(bz-1.5, 2.2), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    ax.set_xticks(zscorex(np.arange(1, n, step=12)))
    ax.set_xticklabels(np.unique(data['year']))
    ax.set_yticks(zscorey(np.arange(150, 301, step=25)))
    ax.set_yticklabels(np.arange(150, 301, step=25))
    ax.set_xlabel('Time')
    ax.set_ylabel('Std ACE rate x 10,000')
axes[2].set_title('Standardized effect size (p = {:0.3f})'.format(gprdd.discPval(b=bz)))


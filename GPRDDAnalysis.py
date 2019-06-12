# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:57:08 2019

@author: Max
"""

import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rc
rc('font',size=12)
rc('font',family='serif')
rc('axes',labelsize=10)

__version__ = "0.0.3"

class ContinuousModel():
    
    isOptimized = False
    def __init__(self, x, y, kernel):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.kernel = kernel.copy()
        self.m = GPy.models.GPRegression(x, y, self.kernel) 
        self.ndim = np.ndim(x)
        self.BICscore = None
    #
    
    def train(self, num_restarts=10, verbose=False):
        # We optimize with some restarts to avoid local minima
        self.m.optimize_restarts(num_restarts=10, verbose=verbose)
        self.isOptimized = True
    #
    
    def predict(self, x_test):
        if len(x_test.shape) == 1:
            x_test = np.atleast_2d(x_test).T
        return self.m.predict(x_test, kern=self.m.kern.copy())
    #
    def BIC(self):
        if not self.isOptimized:
            print('Parameters have not been optimized')
            self.train()
        
        if self.BICscore is None:
            k = self.m.num_params
            L = self.m.log_likelihood()
            BIC = np.log(self.n)*k - 2*L
            self.BICscore = BIC
        return self.BICscore
    #
    
    def plot(self, x_test, axis=None, color=None):
        if axis is None:
            axis = plt.gca()
            
        if color is None:
            color = 'green'
            
        mu, Sigma2 = self.predict(x_test) 
        Sigma = np.sqrt(Sigma2)
            
        if self.kernel.input_dim == 1:
            axis.plot(x_test, mu, label='Optimized prediction', color=color)
            axis.fill_between(x_test, np.squeeze(mu - 0.5*Sigma),np.squeeze(mu + 0.5*Sigma), alpha=0.2, color=color)
        elif self.kernel.input_dim == 2:
            p = int(np.sqrt(x_test.shape[0]))
            x0 = np.reshape(x_test[:,0], newshape=(p,p))
            x1 = np.reshape(x_test[:,1], newshape=(p,p))
            mu_res = np.reshape(mu, newshape=(p,p))
            axis.plot_surface(X=x0, Y=x1, Z=mu_res, color=color, antialiased=True, alpha=0.5, linewidth=0)
            
            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
        else:
            raise('Dimensionality not implemented')
                
    
    
class DiscontinuousModel():
    
    isOptimized = False
    def __init__(self, x, y, kernel, labelFunc):
        
        self.ndim = np.ndim(x)
        if self.ndim==2 and x.shape[1]==1:
            self.ndim=1
        self.labelFunc = labelFunc
        lab1 = labelFunc(x)
        lab2 = np.logical_not(lab1)
        
        # ugly Numpy behaviour
        x1 = x[lab1,]
        x2 = x[lab2,]
        y1 = y[lab1,]
        y2 = y[lab2,]
        
        if len(x1.shape)==1:
            x1 = np.expand_dims(x1, axis=1)
        if len(x2.shape)==1:
            x2 = np.expand_dims(x2, axis=1)
        if len(y1.shape)==1:
            y1 = np.expand_dims(y1, axis=1)
        if len(y2.shape)==1:
            y2 = np.expand_dims(y2, axis=1)
            
        self.models = list()
        self.models.append(ContinuousModel(x1, y1, kernel))
        self.models.append(ContinuousModel(x2, y2, kernel))
        self.BICscore = None
    #
    
    def train(self):
        for model in self.models:
            model.train()
        self.isOptimized = True        
    #
    
    def predict(self, x_test):
        lab1 = self.labelFunc(x_test)
        lab2 = np.logical_not(lab1)
        x1 = np.expand_dims(x_test[lab1,], axis=1)
        x2 = np.expand_dims(x_test[lab2,], axis=1)
        return self.models[0].predict(x1), self.models[1].predict(x2)
    #
    def BIC(self):
        if not self.isOptimized:
            print('Parameters have not been optimized')
            self.train()
        if self.BICscore is None:
            BIC = 0
            for i, model in enumerate(self.models):
                n = model.n
                k = model.m.num_params
                L = model.m.log_likelihood()
                BIC += np.log(n)*k - 2*L
            self.BICscore = BIC
        return self.BICscore
    #
    
    def plot(self, x_test, axis=None, colors=None):
        if axis is None:
            axis = plt.gca()
            
        lab1 = self.labelFunc(x_test)
        lab2 = np.logical_not(lab1)
        x1 = x_test[lab1,]
        x2 = x_test[lab2,]
            
        if self.ndim==1:
            self.models[0].plot(x1, axis=axis, color='red')
            self.models[1].plot(x2, axis=axis, color='blue')
        elif self.ndim==2:
            
            mu1, _ = self.models[0].predict(x1)
            mu2, _ = self.models[1].predict(x2)
            
            p = int(np.sqrt(x_test.shape[0]))
            mu1_aug = np.zeros((p*p,1))
            mu1_aug.fill(np.nan)
            mu1_aug[lab1,] = mu1
            mu1_aug = np.reshape(mu1_aug, newshape=(p,p))
            
            mu2_aug = np.zeros((p*p,1))
            mu2_aug.fill(np.nan)
            mu2_aug[lab2,] = mu2
            mu2_aug = np.reshape(mu2_aug, newshape=(p,p))
            
            x0 = np.reshape(x_test[:,0], newshape=(p,p))
            x1 = np.reshape(x_test[:,1], newshape=(p,p))
            axis.plot_surface(X=x0, Y=x1, Z=mu1_aug, color='red', antialiased=True, alpha=0.5, linewidth=0)
            axis.plot_surface(X=x0, Y=x1, Z=mu2_aug, color='blue', antialiased=True, alpha=0.5, linewidth=0)
            
            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
        else:
            raise('Dimensionality not implemented')


class GPRDDAnalysis():
    
    isOptimized = False
    log_BF_10 = None
    def __init__(self, x, y, kernel, labelFunc):
        self.x = x
        self.y = y
        self.ndim = np.ndim(x)
        self.labelFunc = labelFunc
        
        if np.ndim(x) == 1 and len(x.shape) == 1:
            x = np.atleast_2d(x).T
        
        if len(y.shape) == 1:
            y = np.atleast_2d(y).T        
        
        self.CModel = ContinuousModel(x, y, kernel)
        self.DModel = DiscontinuousModel(x, y, kernel, labelFunc)
    #
    
    def train(self):
        self.CModel.train()
        self.DModel.train()
        self.isOptimized = True
    #    
        
    def predict(self, x_test):
        return self.CModel.predict(x_test), self.DModel.predict(x_test)
    #
    
    def logBayesFactor(self):
        if not self.isOptimized:
            self.train()      
        if self.log_BF_10 is None:
            BIC_cont = self.CModel.BIC()   
            BIC_disc = self.DModel.BIC()        
            self.log_BF_10 = BIC_cont - BIC_disc 
        return self.log_BF_10
    #
    
    def plot(self, x_test):
        
        if self.ndim == 1:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,6), sharex=True, sharey=True)
            lab1 = self.labelFunc(self.x)
            lab2 = np.logical_not(lab1)
            ax1.plot(self.x[lab1], self.y[lab1], linestyle='', marker='o', color='k')
            ax1.plot(self.x[lab2], self.y[lab2], linestyle='', marker='x', color='k')
            self.CModel.plot(x_test, ax1)
            ax1.axvline(x = 0, color='black', linestyle=':')
            ax1.set_title('Continuous model')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax2.plot(self.x[lab1], self.y[lab1], linestyle='', marker='o', color='k')
            ax2.plot(self.x[lab2], self.y[lab2], linestyle='', marker='x', color='k')
            self.DModel.plot(x_test, ax2, colors=('red', 'blue'))
            ax2.axvline(x = 0, color='black', linestyle=':')
            ax2.set_title('Discontinuous model')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')            
            fig.suptitle(r'GP RDD analysis, log BF10 = {:0.4f}'.format(self.logBayesFactor()))
        elif self.ndim == 2:
            fig = plt.figure(figsize=(14,6))
            
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            lab1 = self.labelFunc(self.x)
            lab2 = np.logical_not(lab1)
            ax.scatter(self.x[lab1,0], self.x[lab1,1], self.y[lab1,], marker='o', c='black')
            ax.scatter(self.x[lab2,0], self.x[lab2,1], self.y[lab2,], marker='x', c='black')
            self.CModel.plot(x_test, ax)
            ax.set_title('Continuous model')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel('y')
            
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.scatter(self.x[lab1,0], self.x[lab1,1], self.y[lab1,], marker='o', c='black')
            ax.scatter(self.x[lab2,0], self.x[lab2,1], self.y[lab2,], marker='x', c='black')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel('y')
            self.DModel.plot(x_test, ax, colors=('red', 'blue'))
            ax.set_title('Discontinuous model')
            fig.suptitle('GP RDD analysis, log BF10 = {:0.4f}'.format(self.logBayesFactor()))
        else:
            raise('Dimensionality not implemented')
                
        
    

        
    
        
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:48:54 2018

@author: jb2968
"""

# Code found at https://github.com/xiao-he/HSIC/blob/master/HSIC.py


import math

import numpy as np



def centering(K):

    n = K.shape[0]

    unit = np.ones([n, n])

    I = np.eye(n)

    Q = I - unit/n

    

    return np.dot(np.dot(Q, K), Q)



def rbf(X, sigma=None):

    GX = np.dot(X, X.T)

    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T

    if sigma is None:

        mdist = np.median(KX[KX != 0])

        sigma = math.sqrt(mdist)

    KX *= - 0.5 / sigma / sigma

    np.exp(KX, KX)

    return KX



def HSIC(X, Y):

    return np.sum(centering(rbf(X))*centering(rbf(Y)))



if __name__ == '__main__':

    X = np.random.normal(size=(1000,100))

    Y = np.random.normal(scale=2.0,size=(1000, 100))
    
    Y_2=2*X

    print(HSIC(X, Y))

    print(HSIC(X, X))
    
    print(HSIC(X,Y_2))
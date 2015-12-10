# -*- coding: utf-8 -*-
import data_generator
import dimension_reduction
import testutil
#import pca
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from random import expovariate
from sklearn import linear_model

def percentRelativeError(a, b):
    totalerror  = 0
    min_len = min(len(a), len(b))
    for e1, e2 in zip(a[:min_len], b[:min_len]):
        error = abs((e1-e2)/e2)
        totalerror += error
#     print("total error: %.4f" % (totalerror/min_len))
    return totalerror/min_len*100
    
def runXSimulation(n, p, kappa, dstar, t, plotidx):
    np.random.seed(451212)
    XS, X = data_generator.simData(n, p, kappa, dstar)
    
    errs = np.zeros((len(t), 3));
    idx = 0
    for tt in t:
        plt.subplot(4, 3, plotidx*3+idx+1)
        errs[idx, :] = doRunXSimulation(XS, X, dstar, kappa, tt)
        idx = idx + 1
    return errs
def doRunXSimulation(XS, X, dstar, kappa, t):  
    # arsvd
    U,S,V = dimension_reduction.rsvd(X,dstar, t)
    # svd of signal    
    US, SS, VS = linalg.svd(XS)     
    # svd                
    UQ, sQ, VQ = linalg.svd(X)
    plt.plot(range(dstar), S, color='blue')
    plt.plot(range(dstar), SS[:dstar], color='red')
    plt.plot(range(dstar), sQ[:dstar], color='black')     
    plt.title('k = {0}, t = {1}'.format(kappa, t))    
    rand_err = percentRelativeError(S, sQ)
    total_err = percentRelativeError(S, SS) 
    noise_err = percentRelativeError(sQ[:dstar], SS)
    return rand_err, total_err, noise_err
    
    return

def runYSimulation(n, p, kappa, dstar, t, snr):
    np.random.seed(45121242) #451212
    XS, X = data_generator.simData(n, p, kappa, dstar)
    
    errs = np.zeros((len(t), 3))
    idx = 0
    for tt in t:
        errs[idx, :] = doRunYSimulation(X, XS, n, dstar, tt, snr)
        idx = idx + 1
    return errs
def doRunYSimulation(X, XS, n, dstar, t, snr):
    # arsvd
    U,S,V = dimension_reduction.rsvd(X,dstar, t)
    # svd of signal    
    US, SS, VS = linalg.svd(XS)     
    # svd                
    UQ, sQ, VQ = linalg.svd(X)
    
    RPC = np.dot(U, np.diag(S))
    
    UQ, sQ, VQ = linalg.svd(X)

    SQ = np.diag(sQ)[:dstar, :dstar]
    
    noise = np.random.randn(n)
    #S = linalg.svd(noise)[1]
    #k = 1
    v_j_1 = 1#S[S.shape[0]-1]*k # smallest singular value
    beta = np.zeros(n);
    dmax = round(dstar)
    for i in range(dmax):
        v_j = expovariate(1)
        beta[i] = v_j + v_j_1
        v_j_1 = beta[i]
    
    fnorm = np.linalg.norm(beta)
    #print(fnorm)

    allPC = np.dot(UQ, np.diag(sQ))
    #print(allPC)
    allPC2 = np.dot(X, VS.T)
    #print(allPC2)
    PC = np.dot(UQ[:, :dstar], SQ)
    # XQ = np.dot(PC, VQ[:dstar, :]) # reconstructed
    
    beta2 = np.dot(VS.T[:, :n], beta)
    # betaall = np.random.randn(p)
    beta3 = np.random.randn(X.shape[1])
    Y = np.dot(np.dot(US, SS), beta) + fnorm/snr*noise
    
    regr = linear_model.LinearRegression()
    err_all = testutil.calcError(X, Y, regr)
    err_pc = testutil.calcError(PC, Y, regr)
    err_rpc = testutil.calcError(RPC, Y, regr)
    return err_all, err_pc, err_rpc
    
snr = 100
dstar = 40
t = [1, 2, 3]
n = 200
p = 5000
kappa = [1]

#plotidx = 0
#plt.figure(figsize=(12,16))
#for kappa in [0.5, 1, 2, 3]:
#    result = runXSimulation(n, p, kappa, dstar, t, plotidx)
#    idx = 0    
#    for tt in t:
#        row = result[idx]
#        print("$\kappa=%.1f, t =%d$ & %0.3f & %0.3f & %0.3f\\\\" % (kappa, tt, row[0], row[1], row[2]))
#        idx = idx+1
#    plotidx = plotidx +1    
#plt.savefig('reg.png', bbox_inches='tight')
#plt.show()

for kappa in [0.5]:
    print(runYSimulation(n, p, kappa, dstar, t, snr))
# runSimulation(n, p, kappa, dstar, snr)

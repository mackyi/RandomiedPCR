import numpy as np
import matplotlib.pyplot as plt

def calcError(X, Y, regr):
    n = Y.shape[0]
    x_train = X[:(0.8*n), :]
    x_test = X[(0.8*n):, :]
    y_train = Y[:(0.8*n)]
    y_test = Y[(0.8*n):] 
    regr.fit(x_train, y_train)
    #print("Residual sum of squares: %.4f"
    #      % np.mean((regr.predict(x_test) - y_test) ** 2))
    #plt.scatter(y_test, regr.predict(x_test), color='blue')
    #plt.xticks(())
    #plt.yticks(())
    # plt.show()# -*- coding: utf-8 -*-
    return np.mean((regr.predict(x_test) - y_test) ** 2)

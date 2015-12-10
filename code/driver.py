#!/usr/bin/python

import data_generator
import dimension_reduction
from scipy import linalg

# example usage

S2, X = data_generator.simData(2000,5000,1,25)
dstar = dimension_reduction.stabilityMeasure(X,200, 5,1)
print(dstar)
dstar = 50
U,S,V = dimension_reduction.rsvd(X,dstar, 1)

U3, S3, V3 = linalg.svd(X)
print(S)
print(S2)
print(len(S))
print(len(S2))
#print(S2[len(S)-1])
#print(S2[len(S)])
#print(S2[50:100])

totalerror  = 0
for e1, e2 in zip(S3[0:len(S)], S):
    error = abs((e1-e2)/e2)
    totalerror += error

print("total error: ")
print(totalerror/len(S))
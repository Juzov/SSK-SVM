import numpy as np
import math as math

def kPrimeN(sx,t,i):
    if(i == 0):
        return 1
    if( min(len(sx),len(t)) < i):
        return 0
    
    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    sumJ = 0
    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(s,t[:j-1],i - 1) * lambdaDecay ** (len(t) - j + 2)

    return lambdaDecay * kPrimeN(s,t,i) + sumJ

    

def kN(sx,t,i):
    if( min(len(sx),len(t)) < i):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    sumJ = 0

    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(sx,t[:j-1],i - 1) * lambdaDecay ** 2

    return kN(s,t,i) + sumJ

stringS = 'car'
stringT = 'cat'
lambdaDecay = 0.5
k = 2

notNormalized = kN(stringS,stringT,k)
print(notNormalized)
stringSKernel = kN(stringS,stringS,k)
stringTKernel = kN(stringT,stringT,k)
normalizedKernel = notNormalized / math.sqrt(stringSKernel * stringTKernel)

print(normalizedKernel)


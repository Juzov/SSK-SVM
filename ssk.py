import numpy as np
import math as math
import nltk as nltk
from nltk.corpus import reuters

def kPrimePrimeN(sx,tz,i):

    if( min(len(sx),len(tz)) < i):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    z = tz[-1]
    t = tz[:-1]
    sumJ = 0

    if(x == z):
        #same last elements
        sumJ += lambdaDecay*(kPrimePrimeN(sx, t, i) + lambdaDecay*kPrimeN(s,t, i-1))
    elif(x != z):
        #different last elements
        sumJ += lambdaDecay*kPrimePrimeN(sx, t, i)
    elif(z == t):
        #t has length 1
        for j in range(0,len(t)):
            if(t[j] == x):
                sumJ += kPrimeN(s,t[:j],i - 1) * lambdaDecay ** (len(t) - j + 1)

    return sumJ

def kPrimeN(sx,t,i):
    if(i == 0):
        return 1
    if( min(len(sx),len(t)) < i):
        return 0
    
    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    return lambdaDecay * kPrimeN(s,t,i) + kPrimePrimeN(sx,t,i)

def kN(sx,t,i):
    if( min(len(sx),len(t)) < i):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]
    sumJ = 0

    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(s,t[:j],i - 1) * (lambdaDecay ** 2)
    return kN(s,t,i) + sumJ

#NOTE: If it doesn't work install nltk and add line nltk.download("reuters")
#documentIDList = reuters.fileids()
#print (reuters.raw(documentIDList[0]))

stringS = 'car'
stringT = 'cat'
stringS = stringS.lower()
stringT = stringT.lower()


lambdaDecay = 0.5
k = 2

# notNormalized = kN(stringS,stringT,k)
# print(notNormalized)
# stringSKernel = kN(stringS,stringS,k)
# stringTKernel = kN(stringT,stringT,k)
# normalizedKernel = notNormalized / math.sqrt(stringSKernel * stringTKernel)

# print(normalizedKernel)
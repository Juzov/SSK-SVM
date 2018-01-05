import numpy as np
import math as math
import nltk as nltk
from nltk.corpus import reuters
import timeit

def kPrimePrimeN(sx,t,i):
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

stringS = 'carasdausdhaisdhisdhaisdhasidhaisdhaiuhdiahdiahsdusa'
stringT = 'cataisdhaiuhdiauhsdiaushashdihdaiuhshdaudahidhammmmm'
lambdaDecay = 0.5
k = 2

stringS = stringS.lower()
stringT = stringT.lower()



start = timeit.timeit()
notNormalized = kN(stringS,stringT,k)
end = timeit.timeit()
print("Running time for sskBis - ", end - start)

stringSKernel = kN(stringS,stringS,k)
stringTKernel = kN(stringT,stringT,k)
normalizedKernel = notNormalized / math.sqrt(stringSKernel * stringTKernel)

print(notNormalized)
print(normalizedKernel)

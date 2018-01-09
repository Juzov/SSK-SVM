import numpy as np
import math as math
import nltk as nltk
from nltk.corpus import reuters
import time

def kPrimePrimeN(sx,tz,i,m):
    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]
    print(s)


    sumJ = 0

        
    if(x == z):
        #same last
        z = tz[-1]
        t = tz[:-1] 
        alpha = lambdaDecay * (kPrimePrimeN(sx, t, i, m - 1))
        beta = lambdaDecay*kPrimeN(s,t, i-1,m - 2)
        return  alpha + beta

    for j in range((len(t) - 1),0):
        if(t[j] == x):
            return (lambdaDecay ** (len(t) - j)) * (kPrimePrimeN(sx,tz,i,m - (len(t) - j)))
   

def kPrimeN(sx,t,i,m):

    if(i == 0):
        return 1
    if( min(len(sx),len(t)) < i):
        return 0
    if(m < 2 * i):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    return lambdaDecay * kPrimeN(s,t,i,m - 1) + kPrimePrimeN(sx,t,i,m)

def kN(sx,t,i,m):
    print(sx)

    if( min(len(sx),len(t)) < i):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]
    sumJ = 0

    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(s,t[:j],i - 1, m - 2) * (lambdaDecay ** 2)
    return kN(s,t,i,m) + sumJ

#NOTE: If it doesn't work install nltk and add line nltk.download("reuters")
#documentIDList = reuters.fileids()
#print (reuters.raw(documentIDList[0]))

#stringS = 'sociolinguistics symposium 12 institute education university london 20 bedford london wc1 thursday 26th march ( mid-day ) saturda'
#stringT = 'sociolinguistics symposium 12 institute education university london 20 bedford london wc1 thursday 26th march ( mid-day ) saturda'
stringS = 'car'
stringT = 'cat'
k = 5
lambdaDecay = 0.5

start = time.time()
notNormalized = kN(stringS,stringT,k,100)
end = time.time()

print("Elapsed time: ", end - start)
print(notNormalized)

# notNormalized = kN(stringS,stringT,k)
# print(notNormalized)
# stringSKernel = kN(stringS,stringS,k)
# stringTKernel = kN(stringT,stringT,k)
# normalizedKernel = notNormalized / math.sqrt(stringSKernel * stringTKernel)

# print(normalizedKernel)

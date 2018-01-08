import numpy as np
import math as math
import nltk as nltk
from nltk.corpus import reuters
import time
from tail_recursion import tail_recursive, recurse
import sys

kprimprimdict = {}
kprimdict = {}
kdict = {}

@tail_recursive
def kPrimePrimeN(sx,tz,i,m):
    global kprimprimdict
    if (sx,tz,i) in kprimprimdict:
        return kprimprimdict[(sx,tz,i)]

    if( min(len(sx),len(tz)) < i):
        kprimprimdict[(sx,tz,i)] = 0
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    z = tz[-1]
    t = tz[:-1]
    sumJ = 0

    if(x == z):
        #same last elements
        sumJ += lambdaDecay*(kPrimePrimeN(sx, t, i,m-1) + lambdaDecay*kPrimeN(s,t, i-1,m-2))
    elif(x != z):
        #different last elements
        count = 0
        for ti in range(len(t)-1, -1, -1):
            count += 1
            if(t[ti] == x):
                sumJ += (lambdaDecay**(count))*kPrimePrimeN(sx, t, i, m-count)

    kprimprimdict[(sx,tz,i)] = sumJ
    return sumJ

@tail_recursive
def kPrimeN(sx,t,i, m):
    global kprimdict
    if (sx,t,i) in kprimdict:
        return kprimdict[(sx,t,i)]

    if(i == 0):
        kprimdict[(sx,t,i)] = 1
        return 1
    if( min(len(sx),len(t)) < i):
        kprimdict[(sx,t,i)] = 0
        return 0
    if(m < (2*i)):
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]

    sumkprim = lambdaDecay * kPrimeN(s,t,i,m-1) + kPrimePrimeN(sx,t,i,m)
    kprimdict[(sx,t,i)] = sumkprim
    return sumkprim

@tail_recursive
def kN(sx,t,i,m):
    global kdict
    if (sx,t,i) in kdict:
        return kdict[(sx,t,i)]
    if( min(len(sx),len(t)) < i):
        kprimdict[(sx,t,i)] = 0
        return 0

    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]
    sumJ = 0

    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(s,t[:j],i - 1, m-2) * (lambdaDecay ** 2)

    sumJ = kN(s,t,i,m) + sumJ
    kprimdict[(sx,t,i)] = sumJ
    return sumJ

def getSSK(s,t,i,m):
    global kprimprimdict
    global kprimdict
    global kdict
    kn = kN(s,t,i,m)
    kprimprimdict = {}
    kprimdict = {}
    kdict = {}
    return kn

#NOTE: If it doesn't work install nltk and add line nltk.download("reuters")
#documentIDList = reuters.fileids()
#print (reuters.raw(documentIDList[0]))

sys.setrecursionlimit(100000)
stringS = 'science is organized knowledge'
stringT = 'wisdom is organized life'
# stringS = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'
# stringT = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'
# stringS = 'cat'
# stringT = 'car'

stringS = stringS.lower()
stringT = stringT.lower()

k = 2
lambdaDecay = 0.5
m = 5

notNormalized = getSSK(stringS, stringT, k, m)


# start = time.time()
# normalized = kN(stringS,stringT,k)
# print(normalized)
# end = time.time()

# print("Elapsed time: ", end - start)

# notNormalized = kN(stringS,stringT,k)
# print(notNormalized)
stringSKernel = kN(stringS,stringS,k, 5)
stringTKernel = kN(stringT,stringT,k, 5)
normalizedKernel = notNormalized / math.sqrt(stringSKernel * stringTKernel)

print(normalizedKernel)

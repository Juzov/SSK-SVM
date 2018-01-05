import numpy as np

def kPrimeN(sx,t,i):
    if(i == 1):
        return 1
    
    global lambdaDecay
    print sx

    x = sx[-1]
    s = sx[:-1]
    
    if( min(len(sx),len(t)) < i):
        return 0


    sumJ = 0
    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(s,t[:j-1],i - 1) * lambdaDecay ** (len(t) - j + 2)

    return lambdaDecay * kPrimeN(s,t,i) + sumJ

    

def kN(sx,t,i):
    global lambdaDecay
    x = sx[-1]
    s = sx[:-1]
    

    if( min(len(sx),len(t)) < i):
        return 0

    sumJ = 0

    for j in range(0,len(t)):
        if(t[j] == x):
            sumJ += kPrimeN(sx,t[:j-1],i - 1) * lambdaDecay ** 2

    return kN(s,t,i) + sumJ

stringS = 'manga'
stringT = 'Danska'


lambdaDecay = 0.5

hi = kN(stringS,stringT,3)
print(hi)


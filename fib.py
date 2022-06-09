#Bifurcation diagram
from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

import time
start_time = time.time()
start=0
finish=1
step=0.0001
x = np.zeros(len(range(0,1001)))
M=np.zeros((10000,130))
q=-1.6
filename="./Latex/LateX images/log/q" + str(q) +".png"
g=1

def bif(arxikes):
    pos = 0
    x[1] = arxikes
    for k in np.arange(start,finish,step):
        for i in range(2,1001):
        #x[i] = k * math.sin(k* math.sinh(q * math.sin(2* x[i - 1]))); #np.sine - np.sinh
            x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic
        M[pos,:]=x[-130:]
        pos += 1
        # print(M[pos,:])
    return(M)
k=np.arange(start,finish,step)
b=bif(0.1)
for i in range(0,130):
    plt.plot(k, b[:,i], '.k',alpha=0.8,ms=0.5,)
c=bif(-0.1)
for i in range(0,130):
    plt.plot(k, c[:,i], '.r',alpha=0.7,ms=0.5,)
d=bif(1)
for i in range(0,130):
    plt.plot(k, d[:,i], '.b',alpha=0.5,ms=0.5,)
e=bif(2)
for i in range(0,130):
    plt.plot(k, e[:,i], '.y',alpha=0.2,ms=0.5,)
f=bif(-0.1)
for i in range(0,130):
    plt.plot(k, f[:,i], '.c',alpha=0.1,ms=0.5,)

#plt.xlim(0.8,0.95)
plt.xlabel("k")
plt.ylabel("x")
plt.savefig(filename,bbox_inches='tight',dpi=144,)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()


    


from cmath import inf
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import jit
import numpy as np
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed

start_time = time.time()

r=np.arange(0,1,0.0001)
X=np.ones((len(r),50))
X1=np.ones((10000,10000))
Y=[None] * len(r)
Y1=[None] * len(r)

#@jit(nopython=True)
def bif(r):
    N=300
    k=np.arange(0,1,0.0001)
    x = np.zeros(len(range(0, N)))
    lyapunov = np.zeros(len(range(0, len(k))))
    l1= np.zeros(len(range(0, len(k))))
    x[0]=0.1
    q=-0.1
    for i in range(1,N):
       x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
       lyapunov += np.log(abs(-3*r*(x[i-1]**2-1)))      #derivative of the equation you calculate 
       l1=lyapunov/N
    return (x[-50:],l1)   
 

x1=(bif(x) for x in r)

if __name__ == '__main__':
    # create and configure the process pool
    with Pool(14) as p:      
        for i,ch in enumerate(p.imap(bif,r,chunksize=714)) :
            Y[i]=ch[:][0]
            Y1[i]=ch[:][1]
            x1=np.ones(len(Y[i]))*r[i]
            x2=np.ones(len(Y1[i]))*r[i]   
            X[i]=x1
            X1[i]=x2

# for i,ch in enumerate(x1): 
#     Y[i]=ch[:][0]
#     #print(ch)
#     #print(Y)
#     Y1[i]=ch[:][1]
#     x1=np.ones(len(Y[i]))*r[i]
#     x2=np.ones(len(Y1[i]))*r[i]   
#     X[i]=x1
#     X1[i]=x2

print("--- %s seconds ---" % (time.time() - start_time)) 

plt.plot(X,Y, ".k", alpha=1, ms=0.8)
plt.plot(X1,Y1, ".r", alpha=1, ms=0.8)

plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.rcParams.update({"text.usetex": True})
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()

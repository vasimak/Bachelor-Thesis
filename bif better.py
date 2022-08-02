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
import ray
import threading 
start_time = time.time()

r=np.arange(0,1,0.0001)
X=[]
Y=[]
X1=[]
Y1=[]
@jit(nopython=True)
def bif(r):
    N=1000
    x = np.zeros(len(range(0, N)))
   
    x[0]=0.1
    q=-0.1
    for i in range(1,N):
       x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
    return (x[-130:])   
 

@jit(nopython=True)
def le(r):
    N=1000
    k=np.arange(0,1,0.0001)
    x = np.zeros(len(range(0, N)))
    lyapunov = np.zeros(len(range(0, len(k))))
    l1= np.zeros(len(range(0, len(k))))
    x[0]=0.1
    q=-0.1
    for i in range(1,N):
       x[i]= r *((1 + x[i - 1]))**2* (2 - x[i - 1]) + q
       lyapunov += np.log(abs(-3*r*(x[i-1]**2-1))) #derivative of the equation you calculate 
       l1=lyapunov/N
    return (l1) 


# for i,ch in enumerate(output) :
#     x2=np.ones(len(ch))*r[i]
#     plt.plot(x2,ch, ".k", alpha=1, ms=1.2)
    # for i,ch in enumerate(output1) :
    #     x2=np.ones(len(ch))*r[i]
    #     plt.plot(x2,ch, ".k", alpha=1, ms=1.2)

# lyapunov=(bif(x) for x in r)
# for i,ch in enumerate(lyapunov) :
#     x2=np.ones(len(ch))*r[i]
#     plt.plot(x2,ch, ".k", alpha=1, ms=1.2)
#plt.rcParams.update({"text.usetex": True})


if __name__ == '__main__':
    # create and configure the process pool
    with Pool(4) as p:
        
        for i,ch in enumerate(p.map(bif,r,chunksize=2500)) :
            x1=np.ones(len(ch))*r[i]
            X.append(x1)
            Y.append(ch)
    print("--- %s seconds ---" % (time.time() - start_time))
    with Pool(12) as p:
        for i,ch in enumerate(p.map(le,r,chunksize=833)) :
            x1=np.ones(len(ch))*r[i]
            X1.append(x1)
            Y1.append(ch)
    print("--- %s seconds ---" % (time.time() - start_time))


plt.plot(X,Y, ".k", alpha=1, ms=1.2)
plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.rcParams.update({"text.usetex": True})
print("--- %s seconds ---" % (time.time() - start_time))

#plt.show()
from cmath import inf

import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import njit
import numpy as np
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed

start_time = time.time()

r=np.arange(0,1,0.0001)
X=np.ones((len(r),130))
X1=np.ones((len(r),len(r)))
Y=[None] * len(r)
Y1=[None] * len(r)

# @jit(nopython=True)
# def combined(r):
#     N=1000
#     k=np.arange(0,1,0.0001)
#     x = np.zeros(len(range(0, N)))
#     lyapunov = np.zeros(len(range(0, len(k))))
#     l1= np.zeros(len(range(0, len(k))))
#     x[0]=0.1
#     q=-0.1
#     for i in range(1,N):
#        x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
#        lyapunov += np.log(np.abs(-3*r*(x[i-1]**2-1)))      #derivative of the equation you calculate 
#        l1=lyapunov/N
#     return (x[-130:],l1)   
 




print("--- %s seconds ---" % (time.time() - start_time)) 

plt.plot(X,Y, ".k", alpha=1, ms=0.8)
plt.plot(X1,Y1, ".r", alpha=1, ms=0.8)
print("--- %s seconds ---" % (time.time() - start_time))
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.rcParams.update({"text.usetex": True})
print("--- %s seconds ---" % (time.time() - start_time))

#plt.show()

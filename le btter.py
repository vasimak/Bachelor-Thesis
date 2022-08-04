from cmath import inf
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import njit
import numba as n
import numpy as np
import multiprocessing
from multiprocessing import Pool
from scipy.misc import derivative
start_time = time.time()

r=np.arange(0,1,0.00001)
X1=[]
Y1=[]     


@njit
def le(r):
    N=1000
    lyapunov =0
    l1= 0
    x=0.1
    q=-0.1
    for i in range(1,N):
        #x = x + r - x**2
        x = r *(1 + x) * (1 + x) * (2 - x) + q 
        #lyapunov += np.log(np.abs(1 - 2*x))
        lyapunov += math.log(np.abs(-3*r*(x**2-1))) #derivative of the equation you calculate 
        l1=lyapunov/N
    return (l1) 
# le=map(le,r)
# print(list(le))
# exit()
# if __name__ == '__main__':
#     with Pool(4) as p:
#             for i,ch in enumerate(p.map(le,r,chunksize=25000)) :
#                 # x1=np.ones(len(str((ch))))*r[i]
#                 X1.append(r[i])
#                 Y1.append(ch)
# print("--- %s seconds ---" % (time.time() - start_time))
for i,ch in enumerate(map(le,r)) :
                # x1=np.ones(len(str((ch))))*r[i]
                X1.append(r[i])
                Y1.append(ch)
print("--- %s seconds ---" % (time.time() - start_time))
# print(X1)
# print(Y1)
# exit()
plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
#plt.rcParams.update({"text.usetex": True})
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()

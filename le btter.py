from cmath import inf
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import jit
import numpy as np
start_time = time.time()

r=np.arange(0,1,0.001)
        
@jit(nopython=True)
def le(r):
    N=1000
    k=np.arange(0,1,0.001)
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
lyapunov=(le(x) for x in r)
for i,ch in enumerate(lyapunov) :
    x2=np.ones(len(ch))*r[i]
    plt.plot(x2,ch, ".k", alpha=1, ms=1.2)
#plt.rcParams.update({"text.usetex": True})
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
lyapunov=(bif(x) for x in r)
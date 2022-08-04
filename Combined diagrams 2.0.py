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

start_time = time.time()

r=np.arange(0,1,0.00001)
X=[]
Y=[]
X1=[]
Y1=[]
@njit
def bif(r):
    N=1000
    x = np.zeros(len(range(0, N)))
    x[0]=0.1
    q=-0.1
    for i in range(1,N):
       x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
    return (x[-130:])   
# @njit
# def bif(r):
#     N=1000
#     x3=[] 
#     x = np.zeros(len(range(0, N)))
#     x=0.1
#     q=-0.1
#     for i in range(1,N):
#        x = r *(1 + x) * (1 + x) * (2 - x) + q 
#        if i>=870: 
#         x3.append(x)      
#     return x3

@njit
def le(r):
    N=1000
    lyapunov =0
    l1= 0
    x=0.1
    q=-0.1
    for i in range(1,N):
        x = r *(1 + x) * (1 + x) * (2 - x) + q 
        lyapunov += math.log(np.abs(-3*r*(x**2-1))) #derivative of the equation you calculate 
        l1=lyapunov/N
    return (l1) 


if __name__ == '__main__':
    # create and configure the process pool
    with Pool(8) as p:
        
        for i,ch in enumerate(p.map(bif,r,chunksize=2500)) :
            x1=np.ones(len(ch))*r[i]
            X.append(x1)
            Y.append(ch)
    print("--- %s seconds ---" % (time.time() - start_time))

    with Pool(8) as p:

        for i,ch in enumerate(p.map(le,r,chunksize=25000)) :
            X1.append(r[i])
            Y1.append(ch)
    print("--- %s seconds ---" % (time.time() - start_time))

plt.style.use('dark_background')    
fig, ax = plt.subplots()
plt.figure(1)
plt.subplot(211)
plt.plot(X,Y, ".w", alpha=1, ms=1.2)
plt.subplot(212)
plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.rcParams.update({"text.usetex": True})
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()



#combined in one def (its slower)


# @njit
# def combined(r):
#     N=1000
#     n=N-130
#     x1=[]
#     k=np.arange(0,1,0.0001)
#     lyapunov =0
#     l1= 0
#     x=0.1
#     q=-0.1
#     for i in range(1,N):
#        x = r *(1 + x) * (1 + x) * (2 - x) + q
#        lyapunov += np.log(np.abs(-3*r*(x**2-1)))      #derivative of the equation you calculate 
#        l1=lyapunov/N
#        if i>=n:  
#         x1.append(x)
#     return (x1,l1)

# if __name__ == '__main__':
#     # create and configure the process pool
#     with Pool(4) as p:      
#         for i,ch in enumerate(p.map(combined,r,chunksize=1250)) :
#             Y[i]=ch[:][0]
#             Y1[i]=ch[:][1]
#             x3=np.ones(len(Y[i]))*r[i]
#             x4=r[i]  
#             X[i]=x3
#             X1[i]=x4
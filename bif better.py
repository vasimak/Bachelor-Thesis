from __future__ import print_function
from cmath import inf
import itertools
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import jit
import numpy as np
import multiprocessing 
multiprocessing.cpu_count()
start_time = time.time()
N=150
q=-0.1

x = np.zeros(len(range(0, N)))

y = np.zeros((10,130))
#r = np.zeros(len(range(0, N)))
r=np.linspace(0,1,3)


def bif(r):
    x[0]=0.1
    for i in range(1,N):
        x[i] = r *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q 
    print(x[-130:])
    return(x[-130:])

gen_exp=(bif(x) for x in r)
x1=enumerate(gen_exp)

print(list(x1))
exit()    
plt.plot(k,ch , ".k", alpha=1, ms=1.2)
plt.rcParams.update({"text.usetex": True})
plt.xlabel("k")
plt.ylabel("x")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.savefig(filename,dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

# Bifurcation diagram


from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import jit
import multiprocessing 
multiprocessing.cpu_count()
start_time = time.time()






# the path where the plots are saved. You can change it with yours.


#filename = "./Latex/LateX images/sine q=" + str(q) + "/g" + str(g) + ".jpg"
# filename="./Latex/LateX images/graphs q21/g" + str(g) +".jpg"
# filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".jpg"


# the path where the the data of parameter k and x are saved


#file_path = "./data3/ok q=" + str(q) + " x=" + str(x[1]) + ".txt"

r=np.arange(0,1,0.0001)
 
        
@jit(nopython=True)
def bif(r):
    N=1000
    x = np.zeros(len(range(0, N)))
    lyapunov = np.zeros(len(range(0, N)))
    x[0]=0.1
    q=-0.1
    for i in range(1,N):
       x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q
    return (x[-130:])



x1=(bif(x) for x in r)
p1 = multiprocessing.Process(target=bif)
p2 = multiprocessing.Process(target=bif)
p3 = multiprocessing.Process(target=bif)
p4 = multiprocessing.Process(target=bif)
p5 = multiprocessing.Process(target=bif)
p6 = multiprocessing.Process(target=bif)
for i,ch in enumerate(x1) :
    x2=np.ones(len(ch))*r[i]
    plt.plot(x2,ch, ".k", alpha=1, ms=1.2)
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()

exit()
with open(file_path, "w+", encoding="utf-8", newline="") as f:
    for i in range(10000):
        for j in range(130):
            if np.any(M[i, j] == np.inf) or np.any(M[i, j] == -np.inf):
                break
            else:
                f.writelines([f"{k[i]}", f"{M[i,j]}\n"])

f.close()
print("--- %s seconds ---" % (time.time() - start_time))

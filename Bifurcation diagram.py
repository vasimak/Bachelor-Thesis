#Bifurcation diagram
from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os

import time
start_time = time.time()
pos = 0
start=0
finish=1
step=0.0001 
x = np.zeros(len(range(0,1001)))
M=np.zeros((10000,130))
q=-1.6
x[1] = 0.1
g=1
#filename="./Latex/LateX images/log/q" + str(q) +"  x="+str(x[1])+".png"
filename="./Latex/LateX images/sine q03/g" + str(g) +".png"
file_path="./data3/ok q=" + str(q) +" x="+str(x[1])+".txt"
a =2.4
g = 9
b = 9

for k in np.arange(start,finish,step):
    for i in range(2,1001):
        #x[i] = k * math.sin(k* math.sinh(q * math.sin(2* x[i - 1]))); #np.sine - np.sinh
        x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic
        #  #renyi
        #x[i] = math.cos(k**q * math.acos(x[i - 1]))  #cheb
        #x[i] = a * abs(math.sin(pow(b,3) * math.pi * x[i - 1])) + (1 - a) * (1 - abs(math.sin(pow(k,3) * math.pi * x[i - 1] * (1 - x[i - 1]))))
        #x[i] = (x[i - 1] * math.exp((k + 9) * (1 - x[i - 1])) - (k + 5) * x[i - 1] * (1 - x[i - 1]))%q
    M[pos,:]=x[-130:]
    pos += 1

k=np.arange(start,finish,step)
print(k)
for i in range(0,130):
    plt.plot(k, M[:,i] ,'.k',alpha=0.2,ms=0.5,)
    print(M[:,i])
#plt.xlim(0.8,0.95)
plt.xlabel("k")
plt.ylabel("x")
plt.savefig(filename,bbox_inches='tight',dpi=144,)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()


exit()
with open(file_path, 'w+',encoding='utf-8', newline='') as f:
    for i in range(10000):
        for j in range(130):
            if np.any(M[i,j]==np.inf) or np.any(M[i,j]==-np.inf):
                break
            else:
                f.writelines([f"{k[i]}",f"{M[i,j]}\n"])
            
f.close()
print("--- %s seconds ---" % (time.time() - start_time))

    


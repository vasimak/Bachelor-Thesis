#Bifurcation diagram
from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os
import matplotlib as mpl
import time

start_time = time.time()
pos = 0
start=0
finish=4.4
step=0.00011
N=1001
k=np.zeros(len(range(0,N)))
x = np.zeros(len(range(0,N)))
i = np.zeros(len(range(0,N)))
M=np.zeros((40000,130))
q=0.9
x[0] =0.1
g=3
#filename="./Latex/LateX images/sine q="+ str(q)+"/g" + str(g) +".jpg"
#filename="./Latex/LateX images/graphs q21/g" + str(g) +".jpg"
filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".jpg"

#file_path="./data3/ok q=" + str(q) +" x="+str(x[1])+".txt"

# def kappa(start,finish,step):
#     for j in np.arange(start,finish,step):
#         np.append(k,j)
#     return k
# k=kappa(0,1,0.1)
# x=x[map(lambda i : k *(1 + x[i - 1]) **2 * (2 - x[i - 1]) + q , k)]

for k in np.arange(start,finish,step):
    for i in range(1,N):
        #x[i] = k * math.sin(k* math.sinh(q * math.sin(2* x[i - 1]))); #np.sine - np.sinh
        #x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic
        x[i] = math.cos(k**q * math.acos(q*x[i - 1])) #cheb
            #x[i] = a * abs(math.sin(pow(b,3) * math.pi * x[i - 1])) + (1 - a) * (1 - abs(math.sin(pow(k,3) * math.pi * x[i - 1] * (1 - x[i - 1]))))
            #x[i] = (x[i - 1] * math.exp((k + 9) * (1 - x[i - 1])) - (k + 5) * x[i - 1] * (1 - x[i - 1]))%q
    M[pos,:]=x[-130:]
    pos += 1

k=np.arange(start,finish,step)

plt.figure()
font = {'size': 45}
plt.rc('font', **font) 
for i in range(0,130):
    plt.plot(k, M[:,i] ,'.k',alpha=1,ms=1.2)
plt.xlim(2.55,2.8)
plt.rcParams.update({'text.usetex' : True})
plt.xlabel("k")
plt.ylabel("x")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920/40, 1080/40)
plt.savefig(filename,dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
#plt.show()


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

    


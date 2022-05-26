import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import time
    
start_time = time.time()
x = np.zeros(len(range(0,2001)))
x2 = np.zeros(len(range(0,2001)))
x1 = np.zeros(len(range(0,2001)))
q=0
x[1] = 0.1
#filename="./q=" + str(q) +".png"
#file_path="./datale/logistic q=" + str(q) +".csv"
start=0
finish=4
dim=0.001
N=2001
M=np.zeros((4000,130))
pos = 0
a = 0.9
g = 9
b = 9
e = 0.000000001
k=np.arange(start,finish,dim)
le2=np.zeros(4000)
for j in range(1,len(k)):
    le=0
    x1[1]=x[2000]
    x2[1]=x[2000]
    for i in range(2,2001):
        x[i] = k[j] * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) +q
        x1[i - 1] = x[i - 1] + e
        x1[i] = k[j] * (1 + x1[i - 1]) * (1 + x1[i - 1]) * (2 - x1[i - 1]) +q
        dist=abs(x[i]-x1[i])
        if dist>0:
            le = le + math.log(dist/e)
    le2[j]=le/2000
    # le2=np.dot(x,x1)
    M[pos,:]=x[-130:]
    pos += 1
# q=np.eye(4000)
# le3= linalg.solve_discrete_lyapunov(le2,q )

    
for i in range(130):
    plt.plot(k,M[:,i] ,'.',color='black',markersize=0.5)


LElist = list(le2)
# print(LElist)

for i in range(len(LElist)-1, 0, -1):
    if LElist[i] < -6:
        LElist.pop(i)
        i=i+1 
k=np.linspace(start,finish,len(LElist))
plt.plot(k,LElist,'.',color='red',markersize=1.5,label = 'LE Map')
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("x,LE")
plt.xlim(0,0.6)
plt.legend()
plt.show()    
print("--- %s seconds ---" % (time.time() - start_time))
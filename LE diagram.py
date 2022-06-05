import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import time
start_time = time.time()
q=-2.1
g=2
filename="./Latex/LateX images/graphs q21/g" + str(g) +".png"
#file_path="./datale/logistic q=" + str(q) +".csv"
start=0
finish=1
dim=0.001
N=2001
def LE(start, finish , dim ):
    e = 0.000000001
    a = 0.9
    g = 9
    b = 9
    r=np.arange(start, finish ,dim )
    x = np.zeros(len(range(0,2001)))
    x[1] = 0.1
    le2=np.zeros(len(range(0,len(r))))
    for j in range(1,len(r)):
        le=0
        for i in range(2,2001):
            x[i] = r[j] * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) +q
        x1 = np.zeros(len(range(0,2001)))
        x1[1]=x[2000]
        x2 = np.zeros(len(range(0,2001)))
        x2[1]=x[2000]
        for i in range(2,2001):
            x1[i] = r[j] * (1 + x1[i - 1]) * (1 + x1[i - 1]) * (2 - x1[i - 1]) +q
            x2[i - 1] = x1[i - 1] + e
            x2[i] = r[j] * (1 + x2[i - 1]) * (1 + x2[i - 1]) * (2 - x2[i - 1]) +q
            dist=abs(x1[i]-x2[i])
            if dist>0:
                le = le + math.log(dist/e)
        
        le2[j]=le/2000
    return(le2)


LE=LE(start,finish,dim)
LElist = list(LE)
for i in range(len(LElist)-1, 0, -1):
    if LElist[i] < -6:
        LElist.pop(i)
        i=i+1 
k=np.linspace(start,finish,len(LElist))
plt.plot(k,LElist,'.',color='black',markersize=1.5)
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("LE")
plt.xlim(0,1)
plt.savefig(filename,bbox_inches='tight')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))


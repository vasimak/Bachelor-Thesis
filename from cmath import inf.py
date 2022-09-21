import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import time
start_time = time.time()

#ilename="./LateX images/graphs q14/g" + str(g) +".png"
#file_path="./datale/logistic q=" + str(q) +".csv"
start=0
finish=4
dim=0.001
N=1001
def LE(start, finish , dim ):
    e = 0.000000001
    a = 0.9
    g = 9
    b = 9
    r=np.arange(start, finish ,dim )
    x = np.zeros(len(range(0,1001)))
    x[1] = 0.1
    le2=np.zeros(len(range(0,len(r))))
    for j in range(1,len(r)):
        le=0
        for i in range(2,1001):
            x[i]=np.mod(r[j]*x[i-1],1)
        x1 = np.zeros(len(range(0,1001)))
        x1[1]=x[1000]
        x2 = np.zeros(len(range(0,1001)))
        x2[1]=x[1000]
        for i in range(2,1001):
            x1[i]=np.mod(r[j]*x1[i-1],1)
            x2[i - 1] = x1[i - 1] + e
            x2[i]=np.mod(r[j]*x2[i-1],1)
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
#plt.savefig(filename,bbox_inches='tight')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#
start=0
finish=4.1
step=0.
def LE(start, finish , step ):
    e = 0.000000001
    a = .9
    g = 9
    b = 9
    r=np.arange(start, finish , step)
    x = np.zeros(len(range(0,451)))
    x[1] = 0.1
    le2=np.zeros(len(range(0,len(r))))

    for j in range(1,len(r)):
        le=0
        for i in range(2,451):
            x[i] = r[j] * x[i - 1] * (1 - x[i - 1])
       #print(x[450])
        x1 = np.zeros(len(range(0,8001)))
        x1[1]=x[450]
        #print(x[450])
        x2 = np.zeros(len(range(0,8001)))
        x2[1]=x[450]
        for i in range(2,8001):
            x1[i] = r[j] * x1[i - 1] * (1 - x1[i - 1])
            x2[i - 1] = x1[i - 1] + e
            x2[i] = r[j] * x2[i - 1] * (1 - x2[i - 1])

            dist=abs(x1[i]-x2[i])
            #print(dist,i)
            if dist>0:
                le = le + math.log(dist/e)
                #print(le)

        le2[j]=le/8000
    return(le2)

k=np.arange()
print(len(k))
print(len(LE(0,4.1,0.1)))
np.where(k==)
plt.plot(k,LE(0,4.1,0.1),'.',color='black',markersize=1.5)
plt.axhline(0)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math
from array import *

q=-1.2
k=0.7625
g=12
filename="./LateX images/graphs q12/g" + str(g) +".png"
N=10**6+1
x = np.zeros(len(range(0, N)))
x[0] = 0
x[1] = 1
for i in range(2,N):
    #x[i] = k * np.sin(pi * np.sinh(pi * sin(pi * x(i - 1)))); #np.sine - np.sinh
    x[i] =k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q #logistic
    #x[i] = (k * x[i - 1]) % 1 #renyi
    #x[i]=1.2*x[i-1]*(1-x[i-1])*(2+x[i-1])+k; #cubic-logistic
    #x[i] = np.cos(k * a * math.acos(x(i - 1))) #cheb
    #x[i] = a * abs(np.sin(pow(b,3) * math.pi * x(i - 1))) + (1 - a) * (1 - abs(sin(k ^ 3 * math.pi * x(i - 1) * (1 - x(i - 1)))))
    #x[i] = x(i - 1) * math.exp((k + 9) * (1 - x(i - 1))) - (k + 5) * x(i - 1) * (1 - x(i - 1))%1
print(x[i])

xpoints=x[300:-1]
ypoints=x[301:]
plt.plot(xpoints,ypoints,'.',color='black',markersize=0.5)
plt.xlabel("x(i)")
plt.ylabel("x(i+1)")
# plt.xlim(0,1)
# plt.ylim(0,1)
plt.savefig(filename,bbox_inches='tight')
plt.show()
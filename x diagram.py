import numpy as np
import matplotlib.pyplot as plt
import math
from array import *

q=0.9
k=2.773
g=11
#filename="./Latex/LateX images/sine q="+ str(q)+"/g" + str(g) +".png"
#filename="./Latex/LateX images/graphs q19/" + str(k) +".png"
filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".png"

N=10**6+1
x = np.zeros(len(range(0, N)))
y= np.zeros(len(range(0, N)))
x[0] = 0
x[1] = 0.1
for i in range(2,N):
    #x[i] = k * math.sin(k* math.sinh(q * math.sin(2* x[i - 1]))); #np.sine - np.sinh
    #x[i] =k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q #logistic
    x[i] = math.cos(k**q * math.acos(q*x[i - 1]))  #cheb
# print(x[i])
# y[0] = 0
# y[1] = -0.1
# for i in range(2,N):
#x[i] = k * np.sin(pi * np.sinh(pi * sin(pi * x(i - 1)))); #np.sine - np.sinh
    # y[i] =k *(1 + y[i - 1]) * (1 + y[i - 1]) * (2 - y[i - 1]) + q #logistic
#x[i] = (k * x[i - 1]) % 1 #renyi
#x[i]=1.2*x[i-1]*(1-x[i-1])*(2+x[i-1])+k; #cubic-logistic
#x[i] = np.cos(k * a * math.acos(x(i - 1))) #cheb
# #     #x[i] = a * abs(np.sin(pow(b,3) * math.pi * x(i - 1))) + (1 - a) * (1 - abs(sin(k ^ 3 * math.pi * x(i - 1) * (1 - x(i - 1)))))
# #     #x[i] = x(i - 1) * math.exp((k + 9) * (1 - x(i - 1))) - (k + 5) * x(i - 1) * (1 - x(i - 1))%1

xpoints=x[300:-1]
ypoints=x[301:]
# x1points=y[300:-1]
# y1points=y[301:]
plt.plot(xpoints,ypoints,'.',color='black',markersize=1)
# plt.plot(x1points,y1points,'.',color='red',markersize=1)

plt.xlabel("x(i)")
plt.ylabel("x(i+1)")
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.savefig(filename,bbox_inches='tight')
plt.show()
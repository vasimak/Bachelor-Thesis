import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os
import pandas as pd
from pyparsing import java_style_comment
pos = 0
start=0
finish=4
dim=1000
q=-0.3
g=3
filename="./LateX images/graphs q03/g" + str(g) +".png"
# #q=" + str(q) +".png"
file_path="./data3/ok  x=0.11.csv"
#q=" + str(q) +".csv"
M=np.zeros((130,dim))

l=0.001
a =0.5
g = 9
b = 9
N = 2001
x = np.zeros(len(range(0,2001)),)
k=0.3
for j in np.linspace(0.1,1.7,dim):
    x[1]= j
    for i in range(2,2001):
        #x[i] = k * math.sin(math.pi * math.sinh(math.pi * math.sin(math.pi * x[i - 1])))+q; #np.sine - np.sinh
        x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic
        #x[i] = k * (1-x[i - 1] % 1) + q #renyi
        #x[i] = math.cos(k * math.acos(x[i - 1])) *q #cheb
        #x[i] = a * abs(math.sin(pow(b,3) * math.pi * x[i - 1])) + (1 - a) * (1 - abs(math.sin(pow(k,3) * math.pi * x[i - 1] * (1 - x[i - 1]))))
        #x[i] = x(i - 1) * math.exp((k + 9) * (1 - x(i - 1))) - (k + 5) * x(i - 1) * (1 - x(i - 1))%1
    M[:,pos]=x[-130:]
    pos += 1   
j=np.linspace(0.1,1.7,dim)
for i in range(130):
    plt.plot(j, M[i,:] ,'.',color='black',markersize=0.1)
#plt.xlim(0.3,0.6)
plt.xlabel("x[1]")
plt.ylabel("x")
plt.savefig(filename,bbox_inches='tight')
plt.show()

# print(k)
exit()
with open(file_path, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(["k","x"])
    for i in range(130):
        for count,ele in enumerate(M[i,:]):
        # writer.writerow(M[i,:])
            writer.writerow([k[count],ele])
f.close()
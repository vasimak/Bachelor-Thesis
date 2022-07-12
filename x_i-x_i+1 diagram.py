#x_i - x_(i+1) diagram


import math

import matplotlib.pyplot as plt
import numpy as np

q = 0.9
k = 2.741
g = 10

# filename="./Latex/LateX images/sine q="+ str(q)+"/g" + str(g) +".png"
# filename="./Latex/LateX images/graphs q19/" + str(k) +".png"
filename = "./Latex/LateX images/cheb q=" + str(q) + "/g" + str(g) + ".png"

N = 10**6 + 1
x = np.zeros(len(range(0, N)))
y = np.zeros(len(range(0, N)))
x[0] = 0
x[1] = 0.1

for i in range(2, N):
    # x[i] = k * math.sin(k* math.sinh(q * math.sin(2* x[i - 1]))); #np.sine - np.sinh
    # x[i] =k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q #logistic
    x[i] = math.cos(k**q * math.acos(q * x[i - 1]))  # cheb


xpoints = x[300:-1]
ypoints = x[301:]
plt.plot(xpoints, ypoints, ".", color="black", markersize=1)
plt.xlabel("x(i)")
plt.ylabel("x(i+1)")
plt.savefig(filename, bbox_inches="tight")
plt.show()

# Bifurcation diagram
import csv
import math
import time
from cmath import inf
from numba import njit
import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()
start = 0
finish = 4.2
step = 0.000105

g = 11
# filename="./Latex/LateX images/graphs q16/g" + str(g) +".png"
#filename = "./Latex/LateX images/sine q=" + str(q) + "/g" + str(g) + ".png"

@njit
def bif(arxikes):
    x = np.zeros(len(range(0, 1001)))
    M = np.zeros((40000, 130))
    q = -0.5
    pos = 0
    x[0] = arxikes
    for k in np.arange(start, finish, step):
        for i in range(1, 1001):
            x[i] = k * math.sin(k * math.sinh(q * math.sin(2 * x[i - 1])))
            # np.sine - np.sinh
            # x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic
        M[pos, :] = x[-130:]
        pos += 1
        # print(M[pos,:])
    return M


k = np.arange(start, finish, step)

plt.figure()
font = {"size": 45}
plt.rc("font", **font)

b = bif(0.1)
for i in range(0, 130):
    plt.plot(k, b[:, i], ".k", alpha=1, ms=1.4)
c = bif(0.5)
for i in range(0, 130):
    plt.plot(k, c[:, i], ".r", alpha=0.7, ms=1)
d = bif(1)
for i in range(0, 130):
    plt.plot(k, d[:, i], ".b", alpha=0.5, ms=0.7)
# z=bif(1.5)
# for i in range(0,130):
#     plt.plot(k, z[:,i], '.m',alpha=0.4,ms=0.7 )
# e=bif(2)
# for i in range(0,130):
#     plt.plot(k, e[:,i], '.y',alpha=0.4,ms=0.7 )
# f=bif(-0.1)
# for i in range(0,130):
#     plt.plot(k, f[:,i], '. c',alpha=0.3,ms=0.7)


# plt.xlim(0.8,0.95)
# plt.text(0,0,"$x_0=-0.1$",fontsize=35, color='cyan')
plt.text(0, 0.3, "$x_0=0.1$", fontsize=35, color="black")
plt.text(0, 0.6, "$x_0=0.5$", fontsize=35, color="red")
plt.text(0, 0.9, "$x_0=1$", fontsize=35, color="blue")
# plt.text(0,1.2,"$x_0=1.5$",fontsize=30,color='magenta')
# plt.text(0,1.5,"$x_0=2$",fontsize=30,color='yellow')


plt.xlabel("k")
plt.ylabel("x")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
#plt.savefig(filename, dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

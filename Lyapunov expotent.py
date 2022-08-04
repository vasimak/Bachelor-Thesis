#Lyapunov expotent


import csv
import math
import time
from numba import njit
import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()


# The iterations of the LE  are determined by this three parameters.



# fixed parameters


N = 1001

X1=[]
Y1=[]


e = 0.000000001


# parameters you can change
g = 5  # Parameter only for the saved plots


# the path where the plots are saved. You can change it with yours.


# filename="./Latex/LateX images/sine q="+ str(q)+"/g" + str(g) +".jpg"
# filename="./Latex/LateX images/graphs q21/g" + str(g) +".jpg"
# filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".jpg"
r=np.arange(0,1,0.00001)

# @jit(nopython=True)
# def le(r):
#     k=np.arange(0,1,0.00001)
#     x = np.zeros(len(range(0, N)))
#     x1 = np.zeros(len(range(0, N)))
#     x2 = np.zeros(len(range(0, N)))
#     le2 = []
#     q  = -0.1  # the main parameter that changed the original maps
#     x[0] = 0.1
#     le = 0
#     for i in range(1, N):
#         # x[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x[i - 1])))
#         x[i] = x[i-1] + r - x[i-1]**2
#         #x[i] = r * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q
#         # x[i] = math.cos(r[j]**q * math.acos(q*x[i - 1]))  #cheb
#     x1[0] = x[N - 1]
#     x2[0] = x[N - 1]
#     for i in range(1, N):
#         # x1[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x1[i - 1])))
#         # x2[i - 1] = x1[i - 1] + e
#         # x2[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x2[i - 1])))
#         x1[i] = x1[i-1] + r - x1[i-1]**2
#         #x1[i] = r * (1 + x1[i - 1]) * (1 + x1[i - 1]) * (2 - x1[i - 1]) + q
#         x2[i - 1] = x1[i - 1] + e
#         #x2[i] = r * (1 + x2[i - 1]) * (1 + x2[i - 1]) * (2 - x2[i - 1]) + q
#         x2[i] = x2[i-1] + r - x2[i-1]**2
#          # x1[i] = math.cos(r[j]**q * math.acos(q*x1[i - 1]))
#         # x2[i - 1] = x1[i - 1] + e
#         # x2[i] = math.cos(r[j]**q * math.acos(q*x2[i - 1]))

#         dist = abs(x1[i] - x2[i])
#         if dist > 0:
#             le += math.log(dist / e)

#     le2.append(le / (1000))
#     return le2
# # le=map(le,r)
# # print(list(le))
# # exit()
# print("--- %s seconds ---" % (time.time() - start_time))

# for i,ch in enumerate(le) :
#                 x1=np.ones(len(ch))*r[i]
#                 X1.append(x1)
#                 Y1.append(ch)
@njit
def LE(start, finish, dim):
    x = np.zeros(len(range(0, N)))
    x1 = np.zeros(len(range(0, N)))
    x2 = np.zeros(len(range(0, N)))
    le2 = np.zeros(len(range(0, len(r))))
    q  = -0.1  # the main parameter that changed the original maps
    x[0] = 0.1
    le = 0
    for j in range(1, len(r)):
        le = 0
        for i in range(1, N):
            # x[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x[i - 1])))
            x[i] = r[j] * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q
            # x[i] = math.cos(r[j]**q * math.acos(q*x[i - 1]))  #cheb
        x1[0] = x[N - 1]
        x2[0] = x[N - 1]
        for i in range(1, N):
            # x1[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x1[i - 1])))
            # x2[i - 1] = x1[i - 1] + e
            # x2[i] = r[j] * math.sin(r[j]* math.sinh(q * math.sin(2* x2[i - 1])))

            x1[i] = r[j] * (1 + x1[i - 1]) * (1 + x1[i - 1]) * (2 - x1[i - 1]) + q
            x2[i - 1] = x1[i - 1] + e
            x2[i] = r[j] * (1 + x2[i - 1]) * (1 + x2[i - 1]) * (2 - x2[i - 1]) + q

            # x1[i] = math.cos(r[j]**q * math.acos(q*x1[i - 1]))
            # x2[i - 1] = x1[i - 1] + e
            # x2[i] = math.cos(r[j]**q * math.acos(q*x2[i - 1]))

            dist = abs(x1[i] - x2[i])
            if dist > 0:
                le += math.log(dist / e)
        le2[j] = le / (N - 1)
    return(le2)
LE = LE(0, 1, 0.00001)
LElist = list(LE)



# remove points under -6 if necessary
# for i in range(len(LElist)-1, 0, -1):
#     if LElist[i] < -6:
#         LElist.pop(i)
#         i=i+1
print("--- %s seconds ---" % (time.time() - start_time))

font = {"size": 45}
plt.rc("font", **font)
plt.figure()
plt.plot(r,LElist, ".k", alpha=1, ms=1.4)
#plt.plot(r,LElist, ".k", alpha=1, ms=1.4)
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("LE")
# plt.xlim(0,4.4)
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
# plt.savefig(filename,dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

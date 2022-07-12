# Bifurcation diagram


from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time


start_time = time.time()


# The iterations of the "for" loops are determined by this three parameters. If you change the steps you must change the M's array aswell*.


start = 0
finish = 4
step = 0.0004


# fixed parameters


N = 1001

pos = 0  # the position of the 2d array which for every position we save the last 130 values of x[i]**

k = np.zeros(len(range(0, N)))
x = np.zeros(len(range(0, N)))
i = np.zeros(len(range(0, N)))
M = np.zeros((10000, 130))    # * change the first value of the M array if you change the steps or the finish parameter


# parameters you can change


q = -0.5  # the main parameter that changed the original maps
x[0] = 0.5  # Initial conditions
g = 5  # Parameter only for the saved plots


# the path where the plots are saved. You can change it with yours.


filename = "./Latex/LateX images/sine q=" + str(q) + "/g" + str(g) + ".jpg"
# filename="./Latex/LateX images/graphs q21/g" + str(g) +".jpg"
# filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".jpg"


# the path where the the data of parameter k and x are saved


file_path = "./data3/ok q=" + str(q) + " x=" + str(x[1]) + ".txt"

# "For" loops , to calculate the three systems and put the values of x[i] inside a 2d np.array.

for k in np.arange(start, finish, step):
    for i in range(1, N):

        x[i] = k * math.sin(k * math.sinh(q * math.sin(2 * x[i - 1])))
        # np.sine - np.sinh

        # x[i] = k *(1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q  #logistic

        # x[i] = math.cos(k**q * math.acos(q*x[i - 1])) #cheb

    M[pos, :] = x[-130:]  # ** here you can see that
    pos += 1

# code for plotting the bif diagram


k = np.arange(start, finish, step)

fig = plt.figure()
font = {"size": 45}
plt.rc("font", **font)
for i in range(0, 130):
    plt.plot(k, M[:, i], ".k", alpha=1, ms=1.2)


plt.rcParams.update({"text.usetex": True})
plt.xlabel("k")
plt.ylabel("x")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
plt.savefig(filename,dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()


# the code for saving the data


with open(file_path, "w+", encoding="utf-8", newline="") as f:
    for i in range(10000):
        for j in range(130):
            if np.any(M[i, j] == np.inf) or np.any(M[i, j] == -np.inf):
                break
            else:
                f.writelines([f"{k[i]}", f"{M[i,j]}\n"])

f.close()
print("--- %s seconds ---" % (time.time() - start_time))

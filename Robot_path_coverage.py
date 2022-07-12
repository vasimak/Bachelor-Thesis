# Code for robot path and coverage


import math
import time

import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()
N = 10**5
k = 0.815
q = -1.6
g = 3
filename = "./Latex/LateX images/log/k/g" + str(g) + str(q) + ".jpg"
x = np.zeros(len(range(0, N)))
y = np.zeros(len(range(0, N)))
X = np.zeros(len(range(0, N)))
Y = np.zeros(len(range(0, N)))
M = np.zeros(len(range(0, N)))
theta = np.zeros(len(range(0, N)))


# chaotic maps
def rs(x1, y1):
    # choose parameters for the two chaotic maps used

    x[0] = x1
    y[0] = y1
    # choose parameters for the robot

    X[0] = 0
    Y[0] = 0
    # theta[1]=0
    L = 0.15
    h = 0.2

    for i in range(1, N):
        x[i] = k * ((1 + x[i - 1]) * (1 + x[i - 1])) * (2 - x[i - 1]) + q
        y[i] = k * ((1 + y[i - 1]) * (1 + y[i - 1])) * (2 - y[i - 1]) + q

        # robot coordinates
        X[i] = X[i - 1] + h * math.cos(theta[i - 1]) * (x[i - 1] + y[i - 1]) / 2
        Y[i] = Y[i - 1] + h * math.sin(theta[i - 1]) * (x[i - 1] + y[i - 1]) / 2
        theta[i] = theta[i - 1] + h * (x[i - 1] - y[i - 1]) / L
        # keep the robot inside the boundaries
        if X[i] >= 40 or X[i] <= 0:
            X[i] = X[i - 1] - h * math.cos(theta[i - 1]) * (x[i - 1] + y[i - 1]) / 2
        if Y[i] >= 40 or Y[i] <= 0:
            Y[i] = Y[i - 1] - h * math.sin(theta[i - 1]) * (x[i - 1] + y[i - 1]) / 2
    return (X, Y)


rs1 = rs(0, 0.1)
# rs2=np.array(rs1, dtype=np.int)
plt.figure()
font = {"size": 45}
plt.rc("font", **font)
plt.plot(X, Y, "--b", alpha=0.8, ms=1)
plt.rcParams.update({"text.usetex": True})
plt.rcParams["agg.path.chunksize"] = 10000
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel("X")
plt.ylabel("Y")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
plt.savefig(filename, dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

# create a zero matrix of appropriate dimansions
# assuming each cell is 0.25x0.25
# cell=40/0.25
I = np.zeros((len(np.arange(0, 160)),) * 2)



for j in range(1, len(X)):

    # gia kathe sintetagmeni X,Y, ipologizw to cell (keli) poy antistoixei
    # sto (X,Y)(i), kai epeita sta mikos 1/3, 1/2 kai 2/3 toy diastimatos metaksi
    # (X,Y)(i) kai (X,Y)(i-1). etsi px an se kapoio iteration to robot
    # kanei megalo 'alma', na ipologistoun k ta endiamesa cells

    if X[j] >= 0 or X[j] <= 40 or Y[j] >= 0 or Y[j] <= 40:
        gridx = math.floor(X[j] / 0.25)
        gridy = math.floor(Y[j] / 0.25)
        I[gridx, gridy] = 1

        # 2/3

        gridx = math.floor((0.3 * X[j - 1] + 0.7 * X[j]) / 0.25)
        gridy = math.floor((0.3 * Y[j - 1] + 0.7 * Y[j]) / 0.25)
        I[gridx, gridy] = 1

        # 1/3

        gridx = math.floor((0.7 * X[j - 1] + 0.3 * X[j]) / 0.25)
        gridy = math.floor((0.7 * Y[j - 1] + 0.3 * Y[j]) / 0.25)
        I[gridx, gridy] = 1
        
        # 1/2

        gridx = math.floor((0.5 * X[j - 1] + 0.5 * X[j]) / 0.25)
        gridy = math.floor((0.5 * Y[j - 1] + 0.5 * Y[j]) / 0.25)
        I[gridx, gridy] = 1
        coverage = np.mean((np.mean(I) * 100))
        
print(coverage)
print("--- %s seconds ---" % (time.time() - start_time))

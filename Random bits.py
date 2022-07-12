import time

import numpy as np

start_time = time.time()
N = 10 * 10**6
x = np.zeros(len(range(0, N)))
stream = np.zeros(len(range(0, N)))
stream2 = np.zeros(len(range(0, N)))
q = -0.1
x[0] = 0.1
# file_path="test q=" + str(q) +" x="+str(x[1])+".txt"
file_path = "test1.txt"
k = 0.525

for i in range(1, N - 1):
    x[i] = k * (1 + x[i - 1]) * (1 + x[i - 1]) * (2 - x[i - 1]) + q
stream = 10**8 * x % 1 >= 0.5


with open(file_path, "w+", encoding="utf-8", newline="") as f:
    for item in stream:
        var = 1 if item else 0
        f.writelines(str(var) + "\n")
        # f.writelines( round(val)+"\n" )
print("--- %s seconds ---" % (time.time() - start_time))

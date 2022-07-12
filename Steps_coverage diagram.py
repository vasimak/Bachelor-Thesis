#Steps coverage


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

g = 2

filename = "./Latex/LateX images/log/steps/g" + str(g) + ".jpg"

y1 = [10.46, 19.58, 23.2, 27.57, 51.25, 63.78]
x1 = [10**5, 5 * 10**5, 7 * 10**5, 10**6, 5 * 10**6, 10**7]
y2 = [15.125, 54.254, 64.840, 81.082, 99.98, 100]
x2 = [10**4, 5 * 10**4, 7 * 10**4, 10**5, 5 * 10**5, 10**6]

font = {"size": 45}
plt.rc("font", **font)
mpl.rc("lines", linewidth=8, linestyle="solid")
plt.plot(x2, y2)
plt.scatter(x2, y2, s=600, zorder=2.5)
plt.rcParams.update({"text.usetex": True})
plt.rcParams["agg.path.chunksize"] = 10000
plt.xlabel("Αριθμός Βημάτων")
plt.ylabel("Ποσοστό Κάλυψης (%)")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
plt.savefig(filename, dpi=40)
# print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

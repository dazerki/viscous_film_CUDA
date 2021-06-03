import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

GSx = [0.78, 0.95]; GSy = [75.17, 89.8]
GSSx = [2.87, 2.88]; GSSy = [104.8, 160.7]
GS2Dx = [2.11, 2.52]; GS2Dy = [125.8, 184.8]

roof1x = [0.01, 0.8, 25.71]; roof1y = [1.1, 89.7, 2870]
roof2x = [0.8, 100]; roof2y = [89.7, 89.7]
roof3x = [25.71, 100]; roof3y = [2870, 2870]

fig, ax = plt.subplots(figsize=(10,6))
mpl.rcParams.update({"text.usetex": True})

ref_plot = ax.plot(roof1x, roof1y, 'k--', label='Rooflines')
ref_plot = ax.plot(roof2x, roof2y, 'k--')
ref_plot = ax.plot(roof3x, roof3y, 'k--')
plt.scatter(GSx, GSy, label='3D')
plt.scatter(GSSx, GSSy, marker = 'D', label='3D simplifié')
plt.scatter(GS2Dx, GS2Dy, marker = 's', label='2D')

ax.set_yscale("log")
ax.set_xscale("log")
plt.grid(True, ls="--")
plt.xlabel("Intensité arithmétique [Flop/byte]")
plt.ylabel("Opérations en virgule flottante [GFlop]")
plt.legend(prop={'size': 12})
plt.savefig("./rooflines_GS.pdf")
plt.show()

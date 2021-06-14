import os
from matplotlib import pyplot as plt
import numpy as np

PATH = "/Users/ntdat/Downloads/Data_collected_on_insta/instagrams"

# Load Binh's distribution
# txt = ""
# with open("/Users/ntdat/Downloads/Data_collected_on_insta/binh_distribution_10_06.txt", "r") as f:
#     txt = f.read()
# oy = np.array(txt.split(",")).astype(np.int)
# oy = list(oy)

# Non load Binh's distribution
oy = []
oy_glass = []
oy_mask = []

for r, d, f in os.walk(PATH):
    if "Glass" in r:
        oy_glass.append(len(f))
        continue
    if "Mask" in r:
        oy_mask.append(len(f))
        continue

    oy.append(len(f))
# Save distribution
#   txt += str(len(f) - 1) + ","
#
# with open("distribution1.txt", "w") as f:
#     f.write(txt)


print(len(oy[:-1]), len(oy_glass))

ox = np.arange(len(oy[:-1]))
oy = np.array(oy[:-1])
oy_glass = np.array(oy_glass)

avg_oy = np.full(shape=len(ox), fill_value=np.average(oy), dtype=np.int)
avg_oy_glass = np.full(shape=len(ox), fill_value=np.average(oy_glass), dtype=np.int)
avg_oy_mask = np.full(shape=len(ox), fill_value=np.average(oy_mask), dtype=np.int)

plt.title("Instagram's data distribution 11/06/2021")

plt.bar(ox, oy, label="Total", alpha=0.5, color='r')
plt.bar(ox, oy_glass, label="Glass", alpha=0.5, color='b')
plt.bar(ox, oy_mask, label="Mask", alpha=0.5, color='g')

plt.text(0, avg_oy[0] + 1, 'AVG Total= ' + str(avg_oy[0]), color='r')
plt.text(0, avg_oy_glass[0] + 1, 'AVG Glass= ' + str(avg_oy_glass[0]), color='b')
plt.text(0, avg_oy_mask[0] + 1, 'AVG Mask = ' + str(avg_oy_mask[0]), color='g')
plt.plot(ox, avg_oy, color="r", label="AVG Total")
plt.plot(ox, avg_oy_glass, color="b", label="AVG Glass")
plt.plot(ox, avg_oy_mask, color="g", label="AVG Mask")

plt.legend()
plt.show()

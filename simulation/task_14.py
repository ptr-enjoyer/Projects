import numpy as np
import matplotlib.pyplot as plt
import math

interval, counts = np.loadtxt(f"../../../Data/Task14.txt", unpack = 1, skiprows=1)

width = interval[1]- interval[0]

plt.bar(interval, counts, width = width)
plt.xlabel(f"Interval between events (\u03BCs)")
plt.ylabel("Counts")
plt.title("Smaller Interval Count Plot")
plt.show()
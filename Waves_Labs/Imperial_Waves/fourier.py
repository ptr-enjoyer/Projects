import numpy as np
import matplotlib.pyplot as plt

filename = "../Data/TASK3_05.csv"
x, y = np.loadtxt(filename, skiprows = 1, unpack = 1)
plt.plot(x, y)
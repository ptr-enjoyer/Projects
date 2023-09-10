import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

interval, counts = np.loadtxt(f"../../../Data/Task13.txt", unpack = 1, skiprows=1)
print(interval)
print(counts)
yerr = np.sqrt(counts)
width = 12
plt.bar(interval, counts, width = width, edgecolor = "black", label = "Interval Plot", yerr = yerr, capsize = 4)
plt.xlabel(f"Interval between events (\u03BCs)")
plt.ylabel("Counts")
plt.title("Interval Count Plot")


def expon(x, A, lam):
    y = A * np.exp(-lam * x)
    return y
A_0 = np.max(counts)
mu_0 = np.mean(counts)
lam_0 = 1 / mu_0
p0 = [A_0, lam_0]
fit, err = curve_fit(expon, interval, counts, p0 = p0)
print(fit[1], np.sqrt(np.diag(err)))

print(1/fit[1])

plt.plot(interval, expon(interval, *fit), color = "red", label = "Expopnential Fit")
plt.legend()

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math
from stats import StatTests
from scipy.optimize import curve_fit
from numpy import diag
from pickle import NONE


count = np.loadtxt(f"../../../Data/task_12.txt")


data, bin_edges = np.histogram(count)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
std = np.sqrt(data)
width = bin_edges[1:] - bin_edges[:-1]


def gaussian(A, mu, sig):
    x_2= bin_centers
    x = np.linspace(0, 2*int(mu), len(range(2*int(mu) + 1)))
    y = A * np.exp(-(x-mu)**2 / sig**2)
    y_2 = A * np.exp(-(x_2-mu)**2 / sig**2)
    return x, y, y_2, x_2

def gauss_2(x, A, mu, sig):
    y = A * np.exp(-(x-mu)**2 / sig**2)
    return y

def range_prod(lo,hi):
    if lo+1 < hi:
        mid = (hi+lo)//2
        return range_prod(lo,mid) * range_prod(mid+1,hi)
    if lo == hi:
        return lo
    return lo*hi

def treefactorial(n):
    if n < 2:
        return 1
    return range_prod(1,n)

def poisson(A, mu):
    "Takes the expectiation value for some data mu and returns a poisson distribution."
    n = np.linspace(0, int(2*int(mu)), len(range(2*int(mu) +1)))
    n = np.array(n)
    x = bin_centers
    
    P = []
    P_1 = []
    for i in n:
        p = A * mu ** i * np.exp(-mu) / treefactorial(i)
        P.append(p)
        
    for q in x:
        p = A * mu ** q * np.exp(-mu) / treefactorial(q)
        P_1.append(p)
    
    return n, P, P_1


fact = 20 / 0.04
p0 = [20, 97, 18]
params, cov = curve_fit(gauss_2, bin_centers, data, p0 = p0)
print(params)
print(np.diag(cov))

plt.plot(poisson(fact, 97.89)[0], poisson(fact, 97)[1], color = "orange", label = "Poisson Fit")
plt.plot(gaussian(20.963, 97.89, 16.5)[0], gaussian(20.963, 97.89, 16.5)[1], color = "red", label = "Gaussian Fit")
plt.bar(bin_centers, data, yerr = std, capsize = 4, width = width, edgecolor = "black", label = "St-90 Count")
plt.xlabel("Count")
plt.ylabel("Number of Cycles")
plt.title("St-90 Count Distribution with 6 Al Sheets")


plt.xlim(70, 130)
plt.grid()
plt.legend()
plt.show()

test_1 = StatTests(data, gaussian(20.963, 97.89, 16.5)[2], 3, std )
test_2 = StatTests(data, poisson(fact, 97.89)[2], 2, std)
print(test_1.reduced_chi_squared_test())
print(test_2.reduced_chi_squared_test())

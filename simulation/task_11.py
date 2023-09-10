import numpy as np
import matplotlib.pyplot as plt
import math

count = np.loadtxt(f"../../../Data/task_11.txt")


data, bin_edges = np.histogram(count)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
std = np.sqrt(data)
width = bin_edges[1:] - bin_edges[:-1]

def poisson(A, mu):
    "Takes the expectiation value for some data mu and returns a poisson distribution."
    n = np.linspace(0, int(2*mu), len(range(2*mu +1)))
    n = np.array(n)
    
    P = []
    for i in n:
        p = A * mu ** i * np.exp(-mu) / math.factorial(int(i))
        P.append(p)
    return n, P

plt.bar(bin_centers, data, yerr = std, capsize = 4, width = width, edgecolor = "black")
plt.plot(poisson(270,19)[0], poisson(270,19)[1], color = "red")
plt.xlabel("Counts")
plt.ylabel("Number of Counts")

plt.title("Poisson Distribution with 9 Al Sheets")
plt.legend(["Histogram for the Sr-90", "Poisson fit"])



plt.grid()
plt.show()

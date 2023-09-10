import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import round

def straight(x, m, c):
    y = m*x + c
    return y




files =["../../../Data/Al_1.txt", "../../../Data/Cu_2.txt"]

for file in files:
    string = file[14:16]
    thkns, counts = np.loadtxt(file, unpack = 1, delimiter = "\t", skiprows = 1)
    yerr = np.sqrt(counts)
    yerr_per = yerr/counts 
    counts = np.log10(counts)
    yerr = counts * yerr_per
    
    beta_n = 11
    gamma_n = beta_n + 1
    
    thkns_1 = thkns[8:beta_n]
    counts_1 = counts[8:beta_n]
    thkns_2 = thkns[gamma_n:]
    counts_2 = counts[gamma_n:]
    
    c_intercept_1 = 6
    c_intercept_2 = 2
    grad_1 = (counts_1[-1] - counts_1[0]) / (thkns_1[-1] - thkns_1[0]) 
    grad_2 = (counts_2[-1] - counts_2[0]) / (thkns_2[-1] - thkns_2[0])
    
    p0_1 = [grad_1, c_intercept_1]
    p0_2 = [grad_2, c_intercept_2] 
    
    params_1, cov_1 = curve_fit(straight, thkns_1, counts_1, p0 = p0_1)
    params_2, cov_2 = curve_fit(straight, thkns_2, counts_2, p0 = p0_2)
    
    err_grad1 = np.sqrt(np.diag(cov_1)[0])
    err_grad2 = np.sqrt(np.diag(cov_2)[0])
    err_c1 = np.sqrt(np.diag(cov_1)[1])
    err_c2 = np.sqrt(np.diag(cov_2)[1])
    
    print(f"Equation for Beta Decay Fit: {string}")
    print(f"y = ({params_1[0]} +- {err_grad1})x + ({params_1[1]} +- {err_c1})\n")
    
    print(f"Equation for the Gamma Decay Fit: {string}")
    print(f"y = ({params_2[0]} +- {err_grad2})x + ({params_2[1]} +- {err_c2})\n")
    
    
    plt.errorbar(thkns, counts, yerr= yerr, capsize = 4, fmt = "x", label = f"{string} Count Data")
    
    x_intercept = (np.log10(15)-params_1[1])/ params_1[0]
    x_1 = np.linspace(0, x_intercept)
    x_2 = np.linspace(0, thkns[-1])
    x_int_round = round(x_intercept, 3)
    
    var_x = err_c1**2 / params_1[0]**2 + params_1[1]**2 *err_grad1**2 / params_1[0]**4
    err_x = np.sqrt(var_x)
    err_x_round = round(err_x, 3)
    
    
    plt.plot(x_1, straight(x_1, *params_1), ls = "-", label = "Beta Decay Fit")
    plt.plot(x_2, straight(x_2, *params_2), ls = "-", label = "Gamma Decay Fit")
    plt.axvline(x_intercept, linestyle = "--", color = "red", label = "Stopping Distance")
    #plt.yscale("log")
    plt.xlabel(f"Thickness of {string} stack (mm)")
    plt.ylabel("Counts")
    plt.title(f"{string} Stopping Distance")
    plt.annotate(f"SD = {x_int_round}+-{err_x_round} mm", (x_intercept-0.53, 1.2))
    plt.grid()
    plt.legend()
    plt.show()
    
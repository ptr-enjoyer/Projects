import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from stats import StatTests
from numpy import round

def straight(x, m, c):
    y = m*x + c
    return y

def get_c(x, y, m_0):
    c_0 = y - m_0 * x
    return c_0

density = 2.71
n_order = 6
m_order = 14
sub = 1

files =["../../../Data/Al_1.txt", "../../../Data/Cu_2.txt"]

for file in files:
    string = file[14:16]
    thkns, counts = np.loadtxt(file, unpack = 1, delimiter = "\t", skiprows = 1)
    yerr = np.sqrt(counts)
    yerr_per = yerr/counts 
    counts = np.log10(counts) -np.log10(15)
    yerr = counts * yerr_per
    N = len(thkns)
    chi = []
    
    
    for n in range(3, N-3):
        for m in range(n+3, N-2):
            beta_weak = counts[:n]
            beta_strong = counts[n:m]
            gamma = counts[m:]
            d_weak = thkns[:n]
            d_strong = thkns[n:m]
            d_gamma = thkns[m:]
            yerr_weak = yerr[:n]
            yerr_strong = yerr[n:m]
            yerr_gamma = yerr[m:]
            
            m_1 = (beta_weak[-1] - beta_weak[0]) / (d_weak[-1] - d_weak[0])
            c_1 = get_c(d_weak[0], beta_weak[0], m_1)
            p_1 = [m_1, c_1]
            
            m_2 = (beta_strong[-1] - beta_strong[0]) / (d_strong[-1] - d_strong[0])
            c_2 = get_c(d_strong[0], beta_strong[0], m_2)
            p_2 = [m_2, c_2]
            
            m_3 = (gamma[-1] - gamma[0]) / (d_gamma[-1] - d_gamma[0])
            c_3 = get_c(d_gamma[0], gamma[0], m_2)
            p_3 = [m_3, c_3]
            
            params_1, cov_1 = curve_fit(straight, d_weak, beta_weak, p0=p_1)
            params_2, cov_2 = curve_fit(straight, d_strong, beta_strong, p0=p_2)
            params_3, cov_3 = curve_fit(straight, d_gamma, gamma, p0=p_3)
            
            y_weak = straight(d_weak, *params_1)
            y_strong = straight(d_strong, *params_2)
            y_gamma = straight(d_gamma, *params_3)
            
            chi_1 = StatTests(beta_weak, y_weak, 2, yerr_weak)
            chi_2 = StatTests(beta_strong, y_strong, 2, yerr_strong)
            chi_3 = StatTests(gamma, y_gamma, 2, yerr_gamma)
            
            chi_weak = chi_1.reduced_chi_squared_test()
            chi_strong = chi_2.reduced_chi_squared_test()
            chi_gamma = chi_3.reduced_chi_squared_test()
            #print(chi_weak, chi_strong, chi_gamma)
            
            
            chi_tot = (chi_weak + chi_strong + chi_gamma) / 3
            """
            if chi_weak < 0.1 or chi_weak > 2:
                chi_tot += 100
            elif chi_strong < 0.1 or chi_strong > 2:
                chi_tot += 100
            elif chi_gamma < 0.5 or chi_gamma >  2:
                chi_tot += 100
            """
                
            #print(chi_tot)
            print(f"n = {n}, m = {m}: red chi = {chi_tot}\n")
            chi.append(chi_tot)
            
            #plt.plot(d_weak, y_weak)
            #plt.plot(d_strong, y_strong)
            #plt.plot(d_gamma, y_gamma)
            
            
            
            
            
      
    chi = np.array(chi)      
    print(chi)       
    
    
    
    
    thkns_weak = thkns[:n_order]
    thkns_strong = thkns[n_order:m_order]
    thkns_gamma = thkns[m_order:]
    
    beta_weak = counts[:n_order]
    beta_strong = counts[n_order:m_order]
    gamma = counts[m_order:]
    
    m_1 = (beta_weak[-1] - beta_weak[0]) / (thkns_weak[-1] - thkns_weak[0])
    c_1 = get_c(thkns_weak[0], beta_weak[0], m_1)
    p_1 =  [m_1, c_1]
    params_1, cov_1 = curve_fit(straight, thkns_weak, beta_weak, p0=p_1)
    
    m_2 = (beta_strong[-1] - beta_strong[0]) / (thkns_strong[-1] - thkns_strong[0])
    c_2 = get_c(thkns_strong[0], beta_strong[0], m_2)
    p_2 =  [m_2, c_2]
    params_2, cov_2 = curve_fit(straight, thkns_strong, beta_strong, p0=p_2)
    
    m_3 = (gamma[-1] - gamma[0]) / (thkns_gamma[-1] - thkns_gamma[0])
    c_3 = get_c(thkns_gamma[0], gamma[0], m_3)
    p_3 =  [m_3, c_3]
    params_3, cov_3 = curve_fit(straight, thkns_gamma, gamma, p0=p_3)
    
    m_2_sig = np.sqrt(np.diag(cov_2)[0])
    c_2_sig = np.sqrt(np.diag(cov_2)[1])
    
    m_3_sig = np.sqrt(np.diag(cov_3)[0])
    c_3_sig = np.sqrt(np.diag(cov_3)[1])
    
    y_1 = straight(thkns_weak, *params_1)
    y_2 = straight(thkns_strong, *params_2)
    y_3 = straight(thkns_gamma, *params_3)
    
    A = np.array([[1, -params_2[0]]
                 ,[1, -params_3[0]]])
    
    b = np.array([params_2[1], params_3[1]])
    
    values = np.linalg.solve(A,b)
    x = values[1]
    y = values[0]
    
    
    c_del = c_3 - c_2
    m_del = m_2 - m_3
    
    c_del_sig = np.sqrt(c_2_sig**2 + c_3_sig**2)
    m_del_sig = np.sqrt(m_2_sig**2 + m_3_sig**2)
    
    x_sig = x * np.sqrt((c_del_sig/c_del)**2 + (m_del_sig/m_del)**2)
    
    thkns_strong_ext = np.linspace(thkns_strong[0], x)
    y_22 = straight(thkns_strong_ext, *params_2)
    print(f"Stopping Distance {string} = {x} +- {x_sig} mm")
    
    x_cm = x / 10
    x_sig_cm = x_sig / 10
    x_sig_per = x_sig / x
    
    
    R = x_cm * density
    E_max = np.sqrt(((R/0.11 + 1)**2 - 1) / 22.4)
    E_max_sig = E_max * x_sig_per
    print(f"E_max of beta: {string} = {E_max} +- {E_max_sig} MeV")
    plt.subplot(1, 2, sub)
    plt.errorbar(thkns, counts, yerr= yerr, capsize = 4, fmt = "x", label = f"{string} Count Data")
    plt.plot(thkns_weak, y_1, label = "Lower Energy Beta")
    plt.plot(thkns_strong_ext, y_22, label = "Higher Energy Beta")
    plt.plot(thkns_gamma, y_3, label = "Gamma")
    plt.axvline(x, ls="-.", color = "purple")
    #plt.annotate(f"d = {x} +- {x_sig},", (x, 0.5))
    plt.fill_between(thkns_weak, y_1+0.05, y_1-0.05, alpha= 0.3, color = "orange")
    plt.fill_between(thkns_strong, y_2+0.05, y_2-0.05, alpha=0.3, color="green")
    plt.fill_between(thkns_gamma, y_3+0.05, y_3-0.05, alpha=0.3, color= "red")
    plt.xlabel(f"Thickness of {string} stack (mm)")
    plt.ylabel("Log(Counts)")
    plt.title(f"{string} Stopping Distance")
    plt.grid()
    plt.legend()
          
    
    n_order += -1
    m_order += 0
    sub += 1 
    
    density = density*89.6/27.1

plt.show() 
    
             
            
            
            
    
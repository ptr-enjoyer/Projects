import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp


filename = "../Data/frequency_amplitude.csv"
null, freq, V_o, V_o_err, V_i, V_i_err = np.loadtxt(filename, delimiter = ",", skiprows = 1, unpack = True)
# All units for voltage are in V and units for frequency is kHz

L = 40
omega = 2*np.pi*freq
k = np.pi * null / L 

def volt_ratio(i):
    """i will be the index"""
    V_out = V_o[i]
    V_out_err = V_o_err[i]
    V_in = V_i[i]
    V_in_err = V_i_err[i]
    
    amp_ratio = V_out / V_in
    amp_err = amp_ratio * np.sqrt((V_out_err/V_out)**2 + (V_in_err/V_in)**2)
    
    return amp_ratio, amp_err

A_ratio = []
A_err = []

for i in range(len(V_o)):
    A = volt_ratio(i)[0]
    err = volt_ratio(i)[1]
    A_ratio.append(A)
    A_err.append(err)
    
def line(x, m, c):
    y = m*x + c
    return y

p0 = [A_ratio[-2]/freq[-2], 140]
    
fit_amp, cov_1 = scp.curve_fit(line, freq[-7:], A_ratio[-7:], p0=p0)
x1 = np.linspace(70, freq[-1] + 5, len(A_ratio))
y1 = line(fit_amp[0], x1 ,fit_amp[1])

freq_cut = -fit_amp[1] / fit_amp[0]
print(fit_amp)
print(freq_cut, np.sqrt(cov_1))


plt.errorbar(freq, A_ratio, yerr=A_err, capsize=4, fmt = "x")
plt.plot(x1, y1, "--")
plt.axvline(freq_cut, linestyle = "--", color = "green")
plt.xlabel("Frequency (kHz)")
plt.grid()
plt.ylabel("Amplitude Ratio")
plt.title("Amplitude Attenuation")
plt.legend(["Extrapolation", "Cut-off frequency", "Data"])
plt.show()

def line_1(x, m):
    y = m*x
    return y
    

p0 = omega[1]/k[1]  
m, cov = scp.curve_fit(line_1 , k[:17], omega[:17], p0=p0)
x= np.linspace(0, 2.5, len(omega))
y = m*x
print(m)
print(cov)


plt.scatter(k, omega, s=2)
plt.scatter(x, y, s=2)
plt.legend(["Disperion", "Straigh Line Approx"])
plt.grid()
plt.xlabel("Wavenumber (sections^-1)")
plt.ylabel("Angular Frequency, \u03C9 (kHz)")
plt.title("Dispersion Relation")
plt.show()


    
n = 1
del_omegas = []
del_ks = []
while n <= len(omega)-1:
    del_omega = omega[n] - omega[n-1]
    del_k = k[n] - k[n-1]
    del_omegas.append(del_omega)
    del_ks.append(del_k)
    n += 1
    
V_group = np.array(del_omegas) / np.array(del_ks)
lamb =2*np.pi/k
V_phase = lamb * freq
plt.scatter(freq, V_phase, s=3 )
plt.scatter(freq[:-1], V_group, s=5)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Velocity (sections/s)")
plt.legend(["V_phase", "V_group"])
plt.title("V_group and V_phase")
plt.grid()
#plt.errorbar(freq, A_ratio, yerr=A_err, capsize=4)
plt.show()

              
            
                  
    


    
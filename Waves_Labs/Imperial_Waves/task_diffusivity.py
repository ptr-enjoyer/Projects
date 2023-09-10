import numpy as np
import matplotlib.pyplot as plt
from Fourier_Tasks import square_wave
import math

t, temp = np.loadtxt("../Data/thermal_4amin.txt", unpack = True, skiprows = 3)

t = t/10 # Convert to seconds
plt.plot(t, temp)
T = (t[-1]- t[0])/4
amp = 100
thickness = 7.76e-3

null, squ_wave = square_wave(100, t[-1], T, len(t))

def harmonics(A, T, n):
    """Finds the terms for the fourier series.
    A = Amplitude
    T = Time Period
    n = order of harmonic
    """
    terms = []
    for i in range(1, n+1):
        term_n = 2 * A / (i * np.pi) * (1 - (-1)**i) * np.sin((2 * np.pi * i) / T * t) 
        terms.append(term_n) 
    
    fourier_approx = sum(terms)
    fourier_approx += A
    fourier_approx = fourier_approx/2
    return fourier_approx

def trans_factor(data, fourier_approx):
    """Calculates amplitude transmission factor"""
    amp_inner = np.max(data)
    amp_outer = np.max(fourier_approx)
    gamma = amp_inner / amp_outer
    return gamma

def phase_lag(data, fourier_approx):
    """Calculates the phase lah between fundemental harmonic and data"""
    i = np.where(t==240)
    i = np.array(i)
    list = i.tolist()
    i = list[0]
    cut_data = data[:i[0]]
    cut_approx = fourier_approx[:i[0]]
    amp_inner = np.max(cut_data)
    amp_outer = np.max(cut_approx)
    pos_data = np.where(cut_data==amp_inner)
    t_data_max = t[pos_data]
    pos_approx = np.where(cut_approx==amp_outer)
    t_approx_max = t[pos_approx]
    phase_diff = np.abs(t_data_max - t_approx_max)
    phi = (2 * np.pi / T) * phase_diff
    return phi

def diffusivity_phase(T, thkness, phase):
    """Calculates diffusivity using phase lag.
    T = period
    thkness = thickness of PFTE
    phase = phase lag
    """
    omega = (2 * np.pi) / T
    D = (omega * thkness**2) / (2 * phase**2)
    print(f"Diffusivity (phase): {D[0]} m^2/s")
    return D

def diffusivity_amp(T, thkness, amp):
    """Calculates diffusivity using transmission factor.
    T = period
    thkness = thickness of PFTE
    amp = transmission factor
    """
    omega = (2 * np.pi) / T
    log = math.log(trans_factor(temp, temp_harm))
    D = (omega * thkness**2) / (2 * log**2)
    print(f"Diffusivity (TF): {D} m^2/s")
    return D

temp_harm = harmonics(amp, T, 1)
phase_diff = phase_lag(temp, temp_harm)
print(phase_diff)


diffusivity_phase(T, thickness, phase_lag(temp, temp_harm))
diffusivity_amp(T, thickness, trans_factor(temp, temp_harm))
test_1 = trans_factor(temp, temp_harm)
print(test_1)
test_2 = phase_lag(temp, temp_harm)
print(test_2)

    


plt.plot(t, squ_wave)
plt.plot(t, temp_harm)
plt.title("Diffusivity")
plt.xlabel("Time (s)")
plt.ylabel(f"Temperature ({chr(176)}C)")
plt.legend(["Inner PTFE", "Outer PTFE", "Fourier"], loc = "upper right")
plt.show()










    
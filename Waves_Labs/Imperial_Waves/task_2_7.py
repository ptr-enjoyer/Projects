from diffusivity import Diffusivity as Diff
import numpy as np
import matplotlib.pyplot as plt

files = ["thermal_1amin.txt", "thermal_2amin.txt", "thermal_4amin.txt"
        , "thermal_6min.txt", "thermal_8min.txt"]

files_1 = ["thermal_2amin.txt"]


def final(n):
    legend = []
    for file in files:
        diff = Diff(file, n)
        x_harmonic = range(1, n+1)
        
        #plt.plot(diff.time, diff.fourier_data(3))
        #plt.plot(diff.time, diff.temp)
        #plt.show()
        
        diff_phase = diff.diff_phase_n(3)[0] * 10e6
        diff_p_err = diff.diff_phase_n(3)[1] * 10e6
        diff_amp = diff.diff_amp_n(3)[0] * 10e6
        diff_a_err = diff.diff_amp_n(3)[1] * 10e6
        
        print(diff_phase)
        print(diff_p_err)
        print(diff_amp)
        print(f"{diff_a_err}\n")
        
        
        plt.plot(x_harmonic, diff_amp, color = "blue")
        plt.plot(x_harmonic, diff_phase, color = "red")
        
        
        name = f"{file[8]} min"
        
    plt.legend(["D_trans", "D_phase"])
    plt.xlabel("nth harmonic")
    plt.ylabel("Diffusivities (mm^2/s)")
    plt.title("Transmission and Phase Lag Diffusivities for Data Sets")
    plt.grid()
    plt.show()
    
    
    
final(3)
    
    

from diffusivity import Diffusivity as Diff
from Imperial_Waves.task_2_3 import file
import matplotlib.pyplot as plt

files = ["thermal_1amin.txt", "thermal_1bmin.txt", "thermal_2amin.txt", "thermal_2bmin.txt"
         , "thermal_4amin.txt", "thermal_4bmin.txt", "thermal_6min.txt", "thermal_8min.txt", "thermal_16min.txt"]

back_envelope = ["thermal_1amin.txt", "thermal_8min.txt" ]
back_envelope_1 = ["thermal_2amin.txt", "thermal_6min.txt"]
diff_comp = back_envelope + back_envelope_1
diff_comp = ["thermal_1amin.txt", "thermal_2amin.txt",  "thermal_6min.txt", "thermal_8min.txt"]
#files.pop(5)
#files.pop(3)


def plot(file_array):
    for file in file_array:
        diff = Diff(file, 1)
        diff.plots()
    
def diffusivity_print(file_array):
    for file in file_array:
        diff = Diff(file, 1)
        diff_PL = diff.diffusivity_phase()
        diff_TF = diff.diffusivity_amp()
        print(diff_PL)
        print(f"{diff_TF}\n")
        
def diff_plot(diff_comp):
    true_diff = 0.124e-6
    periods = []
    diff_PL = []
    diff_TF = []
    diff_PL_err = []
    diff_TF_err = []
    for file in diff_comp:
        diff = Diff(file, 1)
        period = diff.period
        periods.append(period)
        PL = diff.diffusivity_phase()
        TF = diff.diffusivity_amp()
        diff_PL.append(PL[0])
        diff_PL_err.append(PL[1])
        diff_TF.append(TF[0])
        diff_TF_err.append(TF[1])
    
    plt.errorbar(periods, diff_PL, yerr = diff_PL_err, capsize = 4)
    plt.errorbar(periods, diff_TF, yerr = diff_TF_err, capsize = 4)
    plt.axhline(true_diff, color = "red")
    plt.legend(["Accepted Value", "Phase Lag", "Transmission Factor"])
    plt.xlabel("Periods (s)")
    plt.ylabel("Diffusivity (m^2/s)")
    plt.title("Diffusivity with Periods")  
    plt.show()
  
    
    
diff_plot(diff_comp)   
        
        
        
        
        
        
        
#plot(files)
#diffusivity_print(files)
#diffusivity_print(back_envelope)
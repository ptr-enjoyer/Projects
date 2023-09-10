from radiosim import RadioSimulation as RadSim
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
print(cwd)
files = ["0.3Mev_e_beam_5000.txt", "2Mev_e_beam_5000.txt", "electron_beam.txt", "sim_data_2.txt"]
titles = ["0.3 MeV Beta Beam", "2.0 MeV Beta Beam", "Electron Beam", "Simulation"]
files_1 = ["0.3Mev_e_beam_5000.txt", "2Mev_e_beam_5000.txt"]
titles_1 = ["0.3 MeV Beta Beam", "2.0 MeV Beta Beam"]


for i, file in enumerate(files):
    radio = RadSim(file)
    radio.histogram()
    plt.xlabel("Energies (keV)")
    plt.ylabel("Number of Counts")
    plt.title(titles[i])
    plt.show()

for i, file in enumerate(files_1):
    radio = RadSim(file)
    radio.histogram()
    plt.xlabel("Energies (keV)")
    plt.ylabel("Number of Counts")
    plt.title(titles_1[i])

plt.xlim(0,800)    
plt.show()
    

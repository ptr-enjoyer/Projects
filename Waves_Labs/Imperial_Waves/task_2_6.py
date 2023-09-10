from diffusivity import Diffusivity as Diff
import matplotlib.pyplot as plt

file = "thermal_4amin.txt"
diff = Diff(file, 1)
b_n = diff.b_n(3)
print(b_n)

a_n = diff.a_n(3)
print(a_n)

temp_fourier = diff.fourier_data(5)
plt.plot(diff.time, temp_fourier)
plt.plot(diff.time, diff.temp)
plt.show()
lag = diff.phase_lag_n(3)
print(lag)
amp = diff.amplitude_n(3)
print(amp)

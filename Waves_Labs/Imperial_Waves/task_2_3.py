from diffusivity  import Diffusivity as Diff

file = "thermal_4amin.txt"
diff = Diff(file, 1)

PL = diff.phase_lag()
TF = diff.trans_factor()
print(PL)
print(TF)
diff_phase = diff.diffusivity_phase()
diff_trans = diff.diffusivity_amp()
print(diff_phase)
print(diff_trans)
diff.plots()


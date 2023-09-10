import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

halpha_data = np.loadtxt('Halpha_spectral_data.csv', delimiter=',', skiprows = 4)
dist_data = np.loadtxt('Distance_Mpc.txt', delimiter = '\t', skiprows = 1)

ordered_indices_one = halpha_data[:,0].argsort()
sorted_halpha =halpha_data[ordered_indices_one]


ordered_indices_two = dist_data[:,0].argsort()
sorted_dist_data = dist_data[ordered_indices_two] 
#sorts the data in ascending observation number


counter = 0

for i in sorted_dist_data[:,2]:
    if i == 0:
       sorted_dist_data = np.delete(sorted_dist_data, counter, axis = 0)
       sorted_halpha = np.delete(sorted_halpha, counter+1, axis = 0)
    else:
        counter +=1
#Itterates through both the txt file and csv file filtering out data with intrument response of 0


halpha_filt = sorted_halpha
dist_filt = sorted_dist_data


def line(m,x,c):
    equation = m*x + c
    return equation

def gauss(x,A,mew,sig):
    equation = A*np.exp(-((x-mew)**2)/(2*sig**2)) 
    return equation

#mu was given the name of mew as i thought mu would be a proteted name (like lambda)

counter_two = 1

params = []
for j in halpha_filt[1:,1:]:
    m_guess = (np.max(j)-np.min(j)/(halpha_filt[0,-1]-halpha_filt[0,1])) 
    c_guess = halpha_filt[counter_two,1]
    fit = curve_fit(line, halpha_filt[0,1:], j, p0= [m_guess, c_guess])
    params.append(fit[0])
    counter_two+=1

#m_guess automises the graidents
#c_guess automises the y-intercepts
#params is now a list containning arrays of m and c for each set of data

freq = halpha_filt[0,1:]

line_data=[]
for k in params:
    y_points = line(k[0],freq,k[1])
    line_data.append(y_points)

residuals = halpha_filt[1:,1:]-line_data



params_gauss= []
cov_gauss = []
for z in residuals:
    index_ret = np.where(z == np.max(z))
    mew_guess = freq[index_ret]
    mew_guess = float(mew_guess)
    gauss_fit = curve_fit(gauss,freq, z, p0 = [np.max(z),mew_guess,1e13])
    params_gauss.append(gauss_fit[0])
    cov_gauss.append(gauss_fit[1])





    
#errors for mu were calculated



def line_gauss(x,m,c,A,mew,sig):
    equation = m*x + c + A*np.exp(-((x-mew)**2)/(2*sig**2))
    return equation

#After plotting the gaussian curves and fiddling with the parameters a little
#It was found that they were incredibly sesnsitive so I decided to re-optimise
#my parameters by optimising my optimised paramters using a line + guassian 



array_params = np.array(params)
array_params_gauss = np.array(params_gauss)
line_gauss_guess = []
for p in range(0,len(params)): 
    joint_params = np.concatenate((array_params[p], array_params_gauss[p]),axis =0)
    line_gauss_guess.append(joint_params)
    
#joins the guesses for the line and gaussian seperately so that they can be
#run through the line_gaussian function



  
counter_3 = 0
params_line_gauss = []
cov_line_gauss = []
for l in halpha_filt[1:,1:]:
    line_gauss_fit = curve_fit(line_gauss,freq,l, p0 = line_gauss_guess[counter_3])
    params_line_gauss.append(line_gauss_fit[0])
    cov_line_gauss.append(line_gauss_fit[1])
    counter_3 +=1

#The re-optimised paramters
mew_err = []
for v in cov_line_gauss:
    mew_err.append(np.sqrt(v[3,3]))
    




freq_shift =[]

for values in params_line_gauss:
    freq_shift.append(values[3])
    
freq_shift = np.array(freq_shift)
wave_shift = 3e8/freq_shift

#convert frequency to wavelenght using fl = c, where l is wavelength



velocities =3e8*(wave_shift**2-(656.28e-9)**2)/(wave_shift**2 + (656.28e-9)**2)
velocities = velocities/1000

#As the eqaution was rather simple I took the time to rearrange it 
#np.linalg.solve could also do this but for this case rearranging by hand was quicker

distances = dist_filt[0:,1]



fit_vel,cov = np.polyfit(distances, velocities, 1, cov = 1)

#errors is mu were not weighted as every error had equal importance
#The values were not anaomalyses and therefore discrimination against htme is unjustified
#The code below does have a polyift for weighted errors in case the reader engages in this dangerous practice
#However this will be commented out :)

#rigged_fit_vel,rigged_cov = np.polyfit(distances, velocities, 1, w = 1/np.array(mew_err), cov = 1)



poly_vel = np.poly1d(fit_vel)






#for m in range(0, len(residuals)):
   # plt.errorbar(freq,residuals[m,0:], zorder=1)
    #plt.plot(freq,gauss(freq, *params_gauss[m]), zorder=2)
    #plt.show()

plt.errorbar(distances, velocities, fmt = 'x', capsize = 4)
plt.plot(distances, poly_vel(distances))
plt.title("Hubble's Velocity vs Distance")
plt.xlabel('Distances (Mpc)')
plt.ylabel('Recessional Velocity (km/s)')
plt.grid()

plt.show()

gradient = fit_vel[0]
error = np.sqrt(cov[0,0])

print("Hubble's Constant:")
print(f'{gradient} +- {error} (km/s)/Mpc')






    
    
   



        
    




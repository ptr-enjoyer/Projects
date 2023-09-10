import numpy as np
import matplotlib.pyplot as plt
from Fourier_Tasks import square_wave
import math

class Diffusivity:
    """Adresses tasks for diffusivity section of labs."""
    def __init__(self,filename, n):
        """Initiates the variables from a file.
        filename = name of file containing data in txt format.
        """
        t, temp = np.loadtxt(f"../Data/{filename}", unpack = True, skiprows = 3)
        if filename[10] == "6":
            place_hold = int(filename[8] + filename[9])
        else:
            place_hold = int(filename[8])
        self.pla_hld = place_hold
        self.time = t / 10 # Converts to seconds.
        self.temp = temp
        self.period = (self.time[-1]- self.time[0])/ 4
        self.amp = 100
        self.thkness = 7.76e-3 # In meters.
        self.err = 0.05e-3 # In meters,
        self.n = n
        
        "Initiate a square wave."
        square = square_wave(self.amp, self.time[-1], self.period, len(self.time))
        self.sq_wave = square[-1]
        
    def harmonics(self):
        """Gets the fourier approximation for the square wave with n terms.
        """
        terms = [] # List containing the values for each term
        for n_term in range(1, (self.n)+1):
            term_n = 2 * self.amp / (n_term * np.pi) * (1 - (-1)**n_term) \
            * np.sin((2 * np.pi * n_term) / self.period * self.time) 
            terms.append(term_n)
        
        fourier_approx = sum(terms) # Sums the terms to get an approximation of the square wave
        fourier_approx += self.amp
        fourier_approx = fourier_approx/2
        return fourier_approx
    
    def a_n(self, n):
        """Obtain the a_n values for the fourier analysis of the data"""
        delt_t = self.time[1] - self.time[0]
        a_n = []
        for i in range(1, n+1):
            
            cos = np.cos((2*np.pi/self.period)*i*self.time)
            cos_temp_mix = self.temp * cos
            t = self.time[0]
            index = 0
            area_approx = 0
            while t <= self.period:
                cos_temp = cos_temp_mix[index]
                area_strip = delt_t * cos_temp
                area_approx += area_strip
                index += 1
                t = self.time[index]
            a = 2 * area_approx / self.period
            a_n.append(a)
        return a_n
    
    def b_n(self, n):
        """Obtain the a_n values for the fourier analysis of the data"""
        delt_t = self.time[1] - self.time[0]
        b_n = []
        for i in range(1, n+1):
            
            sin = np.sin((2*np.pi/self.period)*i*self.time)
            sin_temp_mix = self.temp * sin
            t = self.time[0]
            index = 0
            area_approx = 0
            while t <= self.period:
                sin_temp = sin_temp_mix[index]
                area_strip = delt_t * sin_temp
                area_approx += area_strip
                index += 1
                t = self.time[index]
            b = (2 / self.period) * area_approx 
            b_n.append(b)
        return b_n
        
    
    def fourier_data(self, n):
        data_fourier = []
        for i in range(1, n+1):
            term_i = self.a_n(n)[i-1] * np.cos((2*np.pi/self.period)*i*self.time) \
            + self.b_n(n)[i-1] * np.sin((2*np.pi/self.period)*i*self.time)
            data_fourier.append(term_i)
        data = sum(data_fourier) + np.mean(self.temp)
        return data
    
    def amplitude_n(self, n):
        amplitude = np.sqrt(np.square(self.a_n(n)) + np.square(self.b_n(n)))
        return amplitude
    
    def trans_factor_n(self, n):
        amp_inner = self.amplitude_n(n)
        amp_outer = np.max(self.harmonics()) - np.min(self.harmonics())
        gamma = amp_inner / amp_outer
        return gamma
    
    def phase_lag_n(self, n):
        lag = -np.arctan2(self.a_n(n), self.b_n(n))
        i = 0
        while i < len(range(n)):
            if lag[i] < 0:
                lag += 2 * np.pi
                break
            i += 1
        """for i, v in enumerate(lag):
            if v < 0:
                lag[i] += 2*np.pi
        """
        return lag
            
    def trans_factor(self):
        """Determines the amplitude transmission factor of the first harmonic and the data."""
        amp_inner = np.max(self.temp) - np.min(self.temp)
        amp_outer = np.max(self.harmonics()) - np.min(self.harmonics())
        gamma = amp_inner / amp_outer
        return gamma
        
    def phase_lag(self):
        """Determines phase lag between the first harmonic and the data."""
        period_whole = round(self.period)
        tuple_period = np.where(self.time==period_whole)
        array_period = np.array(tuple_period)
        list_period = array_period.tolist()
        period_pos = list_period[0]
        cut_data = self.temp[:period_pos[0]]
        cut_fourier = self.harmonics()[:period_pos[0]]
        amp_inner = np.max(cut_data)
        amp_outer = np.max(cut_fourier)
        pos_data = np.where(cut_data==amp_inner)
        t_data_max = self.time[pos_data]
        pos_approx = np.where(cut_fourier==amp_outer)
        t_approx_max = self.time[pos_approx]
        phase_diff = np.abs(t_data_max - t_approx_max)
        phi = (2 * np.pi / self.period) * phase_diff
        return phi[0]

    def diffusivity_phase(self):
        """Calculates diffusivity using phase lag."""
        omega = (2 * np.pi) / self.period 
        D_phase = (omega * self.thkness**2) / (2 * self.phase_lag()**2)
        D_uncer_phase =  (omega * self.thkness / self.phase_lag()) * self.err
        return D_phase, D_uncer_phase
        
    def diffusivity_amp(self):
        """Calculates diffusivity using amplitudes of data and first harmonics."""
        omega = (2 * np.pi) / self.period
        log = math.log(self.trans_factor())
        D_amp = (omega * self.thkness**2) / (2 * log**2)
        D_uncer_amp = (omega * self.thkness / log**2) * self.err
        return D_amp, D_uncer_amp
    
    def diff_phase_n(self, n):
        omega = (2 * np.pi) / self.period
        D_phase = (omega * self.thkness**2) / (2 * self.phase_lag_n(n)**2)
        D_uncer_phase =  (omega * self.thkness / self.phase_lag_n(n)) * self.err
        return D_phase, D_uncer_phase
    
    def diff_amp_n(self, n):
        omega = (2 * np.pi) / self.period
        log = np.log(self.trans_factor_n(n))
        D_amp = (omega * self.thkness**2) / (2 * log**2)
        D_uncer_amp = (omega * self.thkness / log**2) * self.err
        return D_amp, D_uncer_amp
        
    def plots(self):
        """Plots of graphs created for diffusivity."""
        temp_fourier = self.harmonics()
        plt.plot(self.time, self.temp)
        plt.plot(self.time, self.sq_wave)
        plt.plot(self.time, temp_fourier )
        plt.title(f"Diffusivity: {self.pla_hld}min Data")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Temperature ({chr(176)}C)")
        plt.legend(["Inner PTFE", "Outer PTFE", "Fourier"], loc = "upper right")
        plt.show()
        
        
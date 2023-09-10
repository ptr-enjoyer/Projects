
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def square_wave(A, x_lim, T, N):
        """
        Generates a square wave with:
        Amplitude =  A
        Time Scale =  x_lim
        period = T
        number of samples = N
        """
        ang_freq = 2 * np.pi / T # angular frequency of square wave
        t = np.linspace(0, x_lim, N, endpoint = True) 
        sq_wave = A/2 * (1 + signal.square(ang_freq * t)) # generate square wave
        return t, sq_wave
    
if __name__ == '__main__':
    
    
    """
    -----------------
    Task 1.2b)
    -----------------
    """
    
    
        
    square_wave(100, 20, 2*np.pi, 1000) 
    
    
    """
    --------------
    Task 1.2c)
    --------------
    """
    
    def terms(A, x_lim, T, N, n):
        values = []
        t = square_wave(A, x_lim, T, N)[0]
        sq_wav = square_wave(A, x_lim, T, N)[1]
        for i in range(1, n+1):    
            term_n = 2 * A / (i * np.pi) * (1 - (-1)**i) * np.sin((2 * np.pi * i) / T * t) 
            values.append(term_n)
        
        
        print(len(values))    
        y = sum(values)
        y += A
        y = y/2
        plt.plot(t, y)
        plt.plot(t, sq_wav)
        plt.title("Fourier Series for Square Wave")
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.legend(["Fourier", "Square Wave"])
        plt.show()
      
    terms(100, 20, 2*np.pi, 1000, 100)
    
    ang_freq = 1 # angular frequency of square wave
    
    t = np.linspace(0, 20, 1000, endpoint = True) 
    sq_wave = (signal.square(ang_freq * t) + 1) * 50 # generate square wave
    plt.plot(t, sq_wave)
    y = 100/2 + 2 * 100 / np.pi * np.sin(t) + (2*100 / (3 * np.pi)) * np.sin(3*t)
    plt.plot(t, y)
    plt.show()
    
         
    
    
            
        
        
    

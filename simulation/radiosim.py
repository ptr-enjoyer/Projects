import numpy as np
import matplotlib.pyplot as plt

class RadioSimulation:
    """A class for analysing data and displaying data"""
    
    def __init__(self, filename):
        energy, source, axis, thkness = np.loadtxt(f"../../../Data/{filename}", unpack = True, skiprows = 1, delimiter = ",")
        self._energy = energy
        self._source = source
        self._axis = axis
        self._thkness = thkness
        
    def histogram(self): 
        """Creates a historgram for the enrgies deposited in the detector"""
        plt.hist(self._energy, bins = 20, edgecolor = "black")
        
    
        
        
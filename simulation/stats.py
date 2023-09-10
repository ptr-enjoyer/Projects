import numpy as np


class StatTests:
    """A class for statisical tests"""
    
    def __init__(self, meas_data, fit_data, num_args, yerr):
        """Initializes the fnction used to fit the data, the measured data and the fit data,
        degrees of freedom and errors.
        
        Parameter:
        
        meas_data (array): array for the data collected
        
        fit_data (array): array of the fitted values using func 
        
        num_args (int): number paramters in the fitting function
        
        yerr (array): errors on the measured data
        """
        
        self.meas_data = meas_data
        self.fit_data = fit_data
        self.num_args = num_args
        self.yerr = yerr
        
    def chi_squared_test(self):
        """Perfroms a chi-squared test on the fitted data points."""
        
        residuals = np.array(self.fit_data - self.meas_data)
        residuals_sq = residuals**2
        terms = residuals_sq / (np.array(self.yerr))**2
        chi_square = np.sum(terms)
        return chi_square
    
    def reduced_chi_squared_test(self):
        """Returns a reduced chi-squared test. 
        If return value is approx = 1, then fit is suitable."""
        
        dof = len(self.meas_data) - self.num_args
        red_chi_square = self.chi_squared_test() / dof
        return red_chi_square
        
        
        
        
    
         
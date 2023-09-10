import numpy as np
import matplotlib.pyplot as plt

import os
cwd = os.getcwd()
print(cwd)

def area_rect(width, height):
    area = width * height
    return area

def sum_int(filename):
    x1, y1 = np.loadtxt(filename, unpack=True, skiprows=1)
    plt.plot(x1, y1)
    plt.show()
    """
    
    x_val = x1[0]
    while x_val <= x1[-1]:
    """
    area_under = 0
    delta_x = (x1[1] - x1[0])
    for i, x in enumerate(x1):
        y = y1[i]
        A = area_rect(delta_x, y)
        area_under += A
    
    return area_under


file_1 = "../semi_circle_hres.txt"
file_2 = "../semi_circle_lres.txt"

area_high_res = sum_int(file_1)
area_low_res = sum_int(file_2)
print(area_high_res)
print(area_low_res)
        
        
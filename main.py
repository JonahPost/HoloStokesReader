# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:12:45 2021

@author: Jonah Post
"""

import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import matplotlib.pyplot as plt

#  Fill in the filenames of the data you want to use. Make sure it is in the data folder.
EMD_fname = "EMD_T_A=0.10_G=0.3000_better.txt"
RN_fname = "RN_A0.1000_T_B0.0000_P0.3000.txt"

path = "data/"
if __name__ == '__main__':
    EMD = DataSet(model="EMD", fname=path + EMD_fname)
    RN = DataSet(model="RN", fname=path + RN_fname)
    print("Data is imported")

    entropy_temp_plot = QuantityQuantityPlot("temperature", "entropy", EMD, RN, exponential=True)
    resistivity_temp_plot = QuantityQuantityPlot("temperature", "resistivity_xx", EMD, exponential=False,
                                                 polynomial=True)
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:12:45 2021

@author: Jonah Post
"""

import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import matplotlib.pyplot as plt
import src.IO_utils as IO_utils

from datetime import datetime
import os

#  Fill in the filenames of the data you want to use. Make sure it is in the data folder.
EMD_fname = "EMD_T_A_G=0.1000.txt"
RN_fname = "RN_A_T_B0.0000_P0.1000.txt"

path = "data/"

# plots_folder = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# os.mkdir(plots_folder)

if __name__ == '__main__':
    EMD = DataSet(model="EMD", fname=path + EMD_fname)
    RN = DataSet(model="RN", fname=path + RN_fname)

    Anot0_mask = (EMD.lattice_amplitude != 0)
    Tcutoff_mask = (EMD.temperature >= 0.02)

    print("Data is imported")
    plots_list = [
        QuantityQuantityPlot("temperature", "entropy", EMD, exponential=True, quantity_multi_line="lattice_amplitude"),
        QuantityQuantityPlot("temperature", "resistivity_xx", EMD, exponential=True, polynomial=True,
                             quantity_multi_line="lattice_amplitude", mask=Anot0_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "kappa_xx", EMD, exponential=True, polynomial=True,
                             quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "alpha_xx", EMD, exponential=True, polynomial=True,
                             quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "kappabar_xx", EMD, exponential=True, polynomial=True,
                             quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "gamma_L", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask, fname_appendix="with_T_cutoff"),
        QuantityQuantityPlot("temperature", "gamma_L", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask * Tcutoff_mask, fname_appendix="without_T_cutoff"),
        QuantityQuantityPlot("temperature", "energy", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Tcutoff_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "pressure", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Tcutoff_mask * Tcutoff_mask),
        QuantityQuantityPlot("temperature", "energy_pressure_ratio", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Tcutoff_mask, logy=False),
        QuantityQuantityPlot("temperature", "conductivity_xy", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask, logy=False, fname_appendix="without_T_cutoff"),
        QuantityQuantityPlot("temperature", "conductivity_xy", EMD, quantity_multi_line="lattice_amplitude",
                             mask=Anot0_mask*Tcutoff_mask, logy=False, fname_appendix="with_T_cutoff"),
        QuantityQuantityPlot("temperature", "equation_of_state", EMD, quantity_multi_line="lattice_amplitude",
                             mask=np.logical_not(Anot0_mask), logy=False, logx=False, fname_appendix="only_A0"),
        QuantityQuantityPlot("temperature", "equation_of_state", EMD, quantity_multi_line="lattice_amplitude",
                             mask=None, logy=False, logx=False)
    ]
    print("plots are build")
    IO_utils.save(plots_list)
    plt.close()
    # plt.show()

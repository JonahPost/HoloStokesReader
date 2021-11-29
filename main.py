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
EMD_fname = "EMD_T_A_G=0.1000_full.txt"
RN_fname = "RN_A_T_B0.0000_P0.1000_full.txt"

path = "data/"

# plots_folder = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# os.mkdir(plots_folder)

if __name__ == '__main__':
    EMD = DataSet(model="EMD", fname=path + EMD_fname)
    RN = DataSet(model="RN", fname=path + RN_fname)

    Anot0_emd = (EMD.lattice_amplitude != 0)
    Tcutoff_emd = (EMD.temperature > 0.0199)

    Anot0_rn = (RN.lattice_amplitude != 0)
    Tcutoff_rn = (RN.temperature > 0.0199)


    print("Data is imported")
    # a =[
    # QuantityQuantityPlot("temperature", "entropy", EMD, RN, quantity_multi_line="lattice_amplitude",
    #                      exponential=True, polynomial=False, logx=False, logy=False, mask1=emd_mask, mask2=rn_mask),
    # QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, RN, quantity_multi_line="temperature",
    #                      exponential=False, polynomial=False, logx=False, logy=False, mask1=emd_mask, mask2=rn_mask, fname_appendix="titleppendixtest")
    # ]

    # IO_utils.save(a)

    plots_list = [
        QuantityQuantityPlot("temperature", "entropy", EMD, RN, quantity_multi_line="lattice_amplitude", exponential=True, mask1=Tcutoff_emd, mask2=Tcutoff_rn),
        QuantityQuantityPlot("temperature", "resistivity_xx", EMD, quantity_multi_line="lattice_amplitude", polynomial=True, mask1=Tcutoff_emd, mask2=Tcutoff_rn),
        QuantityQuantityPlot("temperature", "resistivity_xx", EMD, RN, quantity_multi_line="lattice_amplitude", mask1=Tcutoff_emd, mask2=Tcutoff_rn),
        QuantityQuantityPlot("temperature", "alpha_xx", EMD, quantity_multi_line="lattice_amplitude", mask1=Anot0_emd).ax1.legend(),
        QuantityQuantityPlot("temperature", "kappa_xx", EMD, quantity_multi_line="lattice_amplitude", mask1=Anot0_emd),
        # QuantityQuantityPlot("temperature", "gamma_L_from_sigma", EMD, quantity_multi_line="lattice_amplitude", mask1=Tcutoff_emd),
        # QuantityQuantityPlot("temperature", "gamma_L_from_alpha", EMD, quantity_multi_line="lattice_amplitude", mask1=Tcutoff_emd),
        # QuantityQuantityPlot("temperature", "gamma_L_from_kappabar", EMD, quantity_multi_line="lattice_amplitude", mask1=Tcutoff_emd),
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", EMD, quantity_multi_line="temperature",
                             mask1=Tcutoff_emd),
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", EMD, quantity_multi_line="temperature",
                             mask1=Tcutoff_emd),
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", EMD, quantity_multi_line="temperature",
                             mask1=Tcutoff_emd),
    ]
    print("plots are build")

    # Uncomment the following line to save the plot
    # IO_utils.save(plots_list)


    plt.show()
    plt.close()


#TODO
# do polyfit via np.polyfit
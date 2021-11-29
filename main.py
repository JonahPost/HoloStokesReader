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

    Anot0_mask_emd = (EMD.lattice_amplitude != 0)
    Tcutoff_mask_emd = (EMD.temperature > 0.0199)
    emd_mask = Anot0_mask_emd*Tcutoff_mask_emd

    Anot0_mask_rn = (RN.lattice_amplitude != 0)
    Tcutoff_mask_rn = (RN.temperature > 0.0199)
    rn_mask = Anot0_mask_rn * Tcutoff_mask_rn


    print("Data is imported")
    a =[
    QuantityQuantityPlot("temperature", "entropy", EMD, RN, quantity_multi_line="lattice_amplitude",
                         exponential=True, polynomial=False, logx=False, logy=False, mask1=emd_mask, mask2=rn_mask),
    QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, RN, quantity_multi_line="temperature",
                         exponential=False, polynomial=False, logx=False, logy=False, mask1=emd_mask, mask2=rn_mask, fname_appendix="titleppendixtest")
    ]

    # IO_utils.save(a)

    # plots_list = [
        # QuantityQuantityPlot("temperature", "entropy", EMD, exponential=True, quantity_multi_line="lattice_amplitude", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "resistivity_xx", EMD, exponential=True, polynomial=True,
        #                      quantity_multi_line="lattice_amplitude", mask=Anot0_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "kappa_xx", EMD, exponential=False, polynomial=True,
        #                      quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "alpha_xx", EMD, exponential=False, polynomial=True,
        #                      quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "kappabar_xx", EMD, exponential=False, polynomial=True,
        #                      quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "gamma_L", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("temperature", "gamma_L", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask * Tcutoff_mask, fname_appendix="without_T_cutoff"),
        # QuantityQuantityPlot("temperature", "energy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "pressure", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask * Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "energy_pressure_ratio", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "conductivity_xy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask, logy=False, fname_appendix="without_T_cutoff"),
        # QuantityQuantityPlot("temperature", "conductivity_xy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask*Tcutoff_mask, logy=False, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("temperature", "equation_of_state", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=np.logical_not(Anot0_mask), logy=False, logx=False, fname_appendix="only_A0"),
        # QuantityQuantityPlot("temperature", "equation_of_state", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=None, logy=False, logx=False),
        # QuantityQuantityPlot("temperature", "free_energy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, logy=False, logx=False, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("temperature", "internal_energy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, logy=False, logx=False, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("temperature", "charge_density", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, logy=False, logx=False, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("temperature", "free_energy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=None, logy=False, logx=False, fname_appendix="without_T_cutoff"),
        # QuantityQuantityPlot("temperature", "internal_energy", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=None, logy=False, logx=False, fname_appendix="without_T_cutoff"),
        # QuantityQuantityPlot("temperature", "charge_density", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=None, logy=False, logx=False, fname_appendix="without_T_cutoff"),
        # QuantityQuantityPlot("temperature", "chem_pot", EMD, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "one_over_mu", EMD, polynomial=True, exponential=True, quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "wf_ratio_s2_over_rho2", EMD, exponential=True,
        #                      quantity_multi_line="lattice_amplitude",
        #                       mask=Tcutoff_mask*Anot0_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "wf_ratio_kappa_over_sigmaT", EMD, exponential=True,
        #                      quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask*Anot0_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "tau_L", EMD,
        #                      quantity_multi_line="lattice_amplitude",
        #                      mask=Tcutoff_mask*Anot0_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("temperature", "gamma_L", EMD,exponential=True, quantity_multi_line="lattice_amplitude",
        #                      mask=Anot0_mask, fname_appendix="with_T_cutoff"),
        # QuantityQuantityPlot("lattice_amplitude", "gamma_L", EMD, quantity_multi_line="temperature",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("lattice_amplitude", "plasma_frequency", EMD, quantity_multi_line="temperature",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("lattice_amplitude", "equation_of_state", EMD, exponential=True, quantity_multi_line="temperature",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
        # QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, exponential=False, quantity_multi_line="temperature",
        #                      mask=Tcutoff_mask, fname_appendix="with_T_cutoff", logx=False, logy=False),
    # ]
    print("plots are build")
    # IO_utils.save(plots_list)


    plt.show()
    plt.close()

# add RN data
# do polyfit, via np.polyfit
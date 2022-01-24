# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:01:07 2022

@author: Jonah Post
"""
from src.utils import DataSet
import src.physics as physics
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.plot_utils import *
from examples.RN_plots.SamMartijn import plot_RN
from datetime import datetime
import os

def main():
    path = "data/"

    RN_fname = "RN_ATSeriesHighA.txt"
    RN_A0_fname = "RN_BTSeriesA0"
    EMD_fname = "EMD_T_A_G=0.1000_3_zonderAhalf.txt"


    def mask_fit(mask, fig, labelprefix=None, fit_xmax=None):
        x = 1 / EMD.lattice_amplitude[mask]
        y = EMD.shear_length[mask]
        x, y = utils.remove_nan(x, y)
        x, y = utils.sort(x, y)
        # popt, pcov = curve_fit(func, x, y)
        popt, pol = utils.pol_fit(x, y, type="linear")
        xmax = x[0]
        if fit_xmax:
            xmax = fit_xmax
        xrange = np.linspace(0, xmax)
        # fig.ax1.plot(xrange, pol(xrange), "--", label=labelprefix+r"{:.2g}+{:.2g}$(1/A)$".format(*popt), c="k")
        fig.ax1.plot(xrange, pol(xrange), linestyle=(0, (1, 1)), c="k", linewidth=.5)
        print(r"${:.2g}+{:.2g}x$".format(*popt))
        return

    def fit_rn(x,y, ax, fit_xmax=None):
        x, y = utils.remove_nan(x, y)
        x, y = utils.sort(x, y)
        # popt, pcov = curve_fit(func, x, y)
        popt, pol = utils.pol_fit(x, y, type="linear")
        xmax = x[0]
        if fit_xmax:
            xmax = fit_xmax
        xrange = np.linspace(0, xmax)
        # fig.ax1.plot(xrange, pol(xrange), "--", label=labelprefix+r"{:.2g}+{:.2g}$(1/A)$".format(*popt), c="k")
        ax.plot(xrange, pol(xrange), linestyle=(0, (1, 1)), c="k", linewidth=.5)
        print(r"${:.2g}+{:.2g}x$".format(*popt))
        return

    EMD = DataSet(model="EMD", fname=path + EMD_fname)
    RN = DataSet(model="RN", fname=path + RN_fname)
    print("Data is imported")
    for dataset in [EMD, RN]:
        print(dataset.model)
        physics.calc_properties(dataset)

    ## Initialize plot
    figure_size = (8, 4)
    fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2, figsize=figure_size)


    ## Built all relevant EMD masks
    Anot0_emd = (EMD.lattice_amplitude != 0)
    emd_custom_mask = np.logical_not((EMD.lattice_amplitude < 0.81) * (EMD.lattice_amplitude > 0.79) * (EMD.temperature < 0.016)
                                + (EMD.lattice_amplitude < 1.01) * (EMD.lattice_amplitude > 0.99) * (EMD.temperature < 0.008)
                                )
    emd_Tmask01 = (EMD.temperature > 0.0099) * (EMD.temperature < 0.0101)
    emd_Tmask02 = (EMD.temperature > 0.0199) * (EMD.temperature < 0.0201)
    emd_Tmask03 = (EMD.temperature > 0.0299) * (EMD.temperature < 0.0301)
    emd_Tmask04 = (EMD.temperature > 0.0399) * (EMD.temperature < 0.0401)
    emd_Tmask05 = (EMD.temperature > 0.0499) * (EMD.temperature < 0.0501)
    emd_Tmask06 = (EMD.temperature > 0.0599) * (EMD.temperature < 0.0601)
    emd_Tmask08 = (EMD.temperature > 0.0799) * (EMD.temperature < 0.0801)
    emd_Tmask10 = (EMD.temperature > 0.0999) * (EMD.temperature < 0.1001)

    emd_Tmask = emd_Tmask01 + emd_Tmask03 + emd_Tmask06 + emd_Tmask10
    emd_highAmask = (EMD.lattice_amplitude > 0.02)


    emd_shearlenghtfig = QuantityQuantityPlot("one_over_A", "shear_length", EMD, quantity_multi_line="temperature",
                                          mask1=emd_highAmask * emd_custom_mask * emd_Tmask, linelabels=True, cbar=False, ax=ax3, figure=fig)

    emd_high_A_mask = (EMD.lattice_amplitude > .4001) * emd_custom_mask
    mask_fit(emd_Tmask01*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.01, ")
    # mask_fit(emd_Tmask02*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.02, ")
    mask_fit(emd_Tmask03 * emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.03, ")
    # mask_fit(emd_Tmask04*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.04, ")
    # mask_fit(emd_Tmask05 * emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.05, ")
    mask_fit(emd_Tmask06*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.06, ")
    # mask_fit(emd_Tmask08*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.08, ")
    mask_fit(emd_Tmask10*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.10, ")
    emd_shearlenghtfig.ax1.hlines(np.pi/np.sqrt(2),0,6,"red", "--", label=r"$\pi/(\mu\sqrt{2})$")
    emd_shearlenghtfig.ax1.legend(loc = "upper left")
    emd_shearlenghtfig.ax1.set_xlim(xmax=5)

    emd_sigmaTfig = QuantityQuantityPlot("temperature", "conductivity_T", EMD, quantity_multi_line="lattice_amplitude",
                                     mask1=Anot0_emd*emd_custom_mask*(EMD.temperature>0.0089), ax=ax1, figure=fig)
    emd_sigmaTfig.ax1.set_ylim(ymin=0, ymax=10)
    emd_sigmaTfig.ax1.set_xlim(xmin=0, xmax=0.10)
    emd_temp, emd_conductivity_T_lim = utils.sort(EMD.temperature, EMD.conductivity_T_limit)
    emd_sigmaTfig.ax1.plot(emd_temp, emd_conductivity_T_lim, "--", color="red", label=r"$\tau_L = 2 \pi^3\tau_{\hbar}$")
    emd_sigmaTfig.ax1.legend()

    plot_RN(ax2, ax4)
    rn_high_A_mask = (EMD.lattice_amplitude > .4001)



    ax2.set_xlim(xmin=0, xmax=0.10)
    ax4.set_xlim(xmin=0, xmax=2)

    ax3.set_ylim(ymin=0, ymax=10)
    ax4.set_ylim(ymin=0, ymax=10)
    ax4.legend()

    fig.tight_layout()

    folder_name = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    os.mkdir(folder_name)
    print("new folder made")
    fig.savefig(folder_name + "/PlanckianUniversality.png")
    print("plots are saved")
    plt.show()
    plt.close()

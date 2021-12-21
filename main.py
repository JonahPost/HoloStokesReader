# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:12:45 2021

@author: Jonah Post
"""

import numpy as np
from numpy import ndarray

from src.utils import DataSet
import src.utils as utils
from src.plot_utils import *
import src.physics as physics
import matplotlib.pyplot as plt
import src.IO_utils as IO_utils
from scipy.optimize import curve_fit

##  Fill in the filenames of the data you want to use. Make sure it is in the data folder.
EMD_fname = "EMD_T_A_G=0.1000_full.txt"
RN_fname = "RN_A_T_B0.0000_P0.1000_full.txt"

path = "data/"

if __name__ == '__main__':
    EMD = DataSet(model="EMD", fname=path + EMD_fname)
    RN = DataSet(model="RN", fname=path + RN_fname)
    print("Data is imported")
    for dataset in [EMD, RN]:
        print(dataset.model)
        physics.calc_properties(dataset)

    ## masks
    Anot0_emd = (EMD.lattice_amplitude != 0)
    Acutoff_emd = (EMD.lattice_amplitude > 0.02001)
    Tcutoff_emd = (EMD.temperature > 0.0199)
    Anot0_rn = (RN.lattice_amplitude != 0)
    Acutoff_rn = (RN.lattice_amplitude > 0.02001)
    Tcutoff_rn = (RN.temperature > 0.0199)


    def func(x, a1, a2):
        return a1 * x + a2 * (x ** 2)


    def mask_fit(mask, fig):
        x = EMD.lattice_amplitude[mask]
        y = EMD.resistivity_xx[mask]
        x, y = utils.remove_nan(x, y)
        x, y = utils.sort(x, y)
        popt, pcov = curve_fit(func, x, y)
        xrange = np.linspace(x[0], x[-1])
        fig.ax1.plot(xrange, func(xrange, *popt), "--", c="k")
        print(r"${:.2g}x+{:.2g}x^2$".format(*popt))
        return


    def plot_resistivity(save=False):
        resfig1 = QuantityQuantityPlot("temperature", "resistivity_xx", EMD, RN,
                                       quantity_multi_line="lattice_amplitude",
                                       mask1=Tcutoff_emd, mask2=Tcutoff_rn)
        resfig2 = QuantityQuantityPlot("temperature", "resistivity_xx", EMD,
                                       quantity_multi_line="lattice_amplitude",
                                       mask1=Tcutoff_emd, polynomial=True)
        resfig3 = QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, RN,
                                       quantity_multi_line="temperature",
                                       mask1=Tcutoff_emd, mask2=Tcutoff_rn, exponential=False)
        resfig4 = QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, quantity_multi_line="temperature",
                                       mask1=Tcutoff_emd)
        T0_02mask = (EMD.temperature < 0.0201) * (EMD.temperature > 0.0199)
        T0_03mask = (EMD.temperature < 0.0301) * (EMD.temperature > 0.0299)
        T0_04mask = (EMD.temperature < 0.0401) * (EMD.temperature > 0.0399)
        T0_05mask = (EMD.temperature < 0.0501) * (EMD.temperature > 0.0499)
        mask_fit(T0_02mask, resfig4)
        mask_fit(T0_03mask, resfig4)
        mask_fit(T0_04mask, resfig4)
        mask_fit(T0_05mask, resfig4)
        if save:
            plots_list = [resfig1, resfig2, resfig3, resfig4]
            IO_utils.save(plots_list)
        return


    def plot_gammaL(save=False):
        fig_sigma = QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        fig_alpha = QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        fig_kappabar = QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        # fig_sigma.ax1.set_ylim(ymax=0.0075)
        # fig_alpha.ax1.set_ylim(ymax=0.0075)
        # fig_kappabar.ax1.set_ylim(ymax=0.0075)
        fig1 = QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_alpha", EMD,
                                    quantity_multi_line="temperature",
                                    mask1=Tcutoff_emd * Anot0_emd)
        fig2 = QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_kappabar", EMD,
                                    quantity_multi_line="temperature",
                                    mask1=Tcutoff_emd * Anot0_emd)
        fig1.ax1.set_ylim(-1, 1)
        fig2.ax1.set_ylim(-1, 1)
        fig3 = QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_alpha", RN,
                                    quantity_multi_line="temperature",
                                    mask1=Tcutoff_rn * Anot0_rn)
        fig4 = QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_kappabar", RN,
                                    quantity_multi_line="temperature",
                                    mask1=Tcutoff_rn * Anot0_rn)
        fig3.ax1.set_ylim(-.4, .025)
        fig4.ax1.set_ylim(-.4, .025)
        if save:
            plots_list = [fig_sigma, fig_alpha, fig_kappabar, fig1, fig2, fig3, fig4]
            IO_utils.save(plots_list)
        return


    def plot_energy_pressure(save=False):
        energyfig = QuantityQuantityPlot("temperature", "energy", EMD, quantity_multi_line="lattice_amplitude")
        pressurefig = QuantityQuantityPlot("temperature", "pressure", EMD, quantity_multi_line="lattice_amplitude")
        pressuredifffig = QuantityQuantityPlot("temperature", "pressurediffxxyy", EMD, quantity_multi_line="lattice_amplitude", mask1=(EMD.lattice_amplitude > 0.30001))
        E_over_P_fig = QuantityQuantityPlot("temperature", "energy_pressure_ratio", EMD,
                                            quantity_multi_line="lattice_amplitude", mask1=Tcutoff_emd,
                                            mask2=Tcutoff_rn)
        E_over_P_fig.ax1.set_xlim(left=-0.004, right=0.105)
        if save:
            IO_utils.save([energyfig, pressurefig, E_over_P_fig])
        return


    def plot_conductivities(save=False):
        sigmaplot = QuantityQuantityPlot("temperature", "conductivity_xx", EMD,
                                         quantity_multi_line="lattice_amplitude",
                                         mask1=Anot0_emd)
        alphaplot = QuantityQuantityPlot("temperature", "alpha_xx", EMD,
                                         quantity_multi_line="lattice_amplitude",
                                         mask1=Anot0_emd)
        kappabarplot = QuantityQuantityPlot("temperature", "kappabar_xx", EMD,
                                            quantity_multi_line="lattice_amplitude",
                                            mask1=Anot0_emd)
        if save:
            IO_utils.save([sigmaplot, alphaplot, kappabarplot])


    def plot_wf_ratio(save=False):
        EMD.wf_diff = EMD.wf_ratio - EMD.s2_over_rho2
        RN.wf_diff = RN.wf_ratio - RN.s2_over_rho2
        wfplot = QuantityQuantityPlot("temperature", "wf_ratio", EMD, RN, quantity_multi_line="lattice_amplitude",
                                      mask1=Anot0_emd, mask2=Anot0_rn)
        s2rho2plot = QuantityQuantityPlot("temperature", "s2_over_rho2", EMD, RN,
                                          quantity_multi_line="lattice_amplitude", mask1=Anot0_emd, mask2=Anot0_rn)
        wfdiffplot = QuantityQuantityPlot("temperature", "wf_diff", EMD, RN, quantity_multi_line="lattice_amplitude",
                                          mask1=Anot0_emd, mask2=Anot0_rn)
        if save:
            IO_utils.save([wfplot, s2rho2plot, wfdiffplot])
        return


    def plot_EoS(save=False):
        EoSplot = QuantityQuantityPlot("temperature", "equation_of_state", EMD, RN,
                                       quantity_multi_line="lattice_amplitude")
        EoSplot_rn = QuantityQuantityPlot("temperature", "equation_of_state", RN,
                                          quantity_multi_line="lattice_amplitude")
        # EoSplot = QuantityQuantityPlot("lattice_amplitude", "equation_of_state", EMD, RN,
        #                                quantity_multi_line="temperature")
        # EoSplot_rn = QuantityQuantityPlot("lattice_amplitude", "equation_of_state", RN,
        #                                   quantity_multi_line="temperature")
        if save:
            IO_utils.save([EoSplot, EoSplot_rn])
        return

    def plot_sigmaQ(save=False):
        fig1 = QuantityQuantityPlot("lattice_amplitude", "sigmaQ_from_sigma_alpha", EMD, RN,
                                    quantity_multi_line="temperature", mask1=Anot0_emd, mask2=Anot0_rn)
        fig2 = QuantityQuantityPlot("lattice_amplitude", "sigmaQ_from_sigma_kappabar", EMD, RN,
                                    quantity_multi_line="temperature", mask1=Anot0_emd, mask2=Anot0_rn)
        fig3 = QuantityQuantityPlot("lattice_amplitude", "sigmaQ_from_alpha_kappabar", EMD, RN,
                                    quantity_multi_line="temperature", mask1=Anot0_emd, mask2=Anot0_rn)
        fig1 = QuantityQuantityPlot("temperature", "sigmaQ_from_sigma_alpha", EMD, RN,
                                    quantity_multi_line="lattice_amplitude", mask1=Anot0_emd, mask2=Anot0_rn)
        fig2 = QuantityQuantityPlot("temperature", "sigmaQ_from_sigma_kappabar", EMD, RN,
                                    quantity_multi_line="lattice_amplitude", mask1=Anot0_emd, mask2=Anot0_rn)
        fig3 = QuantityQuantityPlot("temperature", "sigmaQ_from_alpha_kappabar", EMD, RN,
                                    quantity_multi_line="lattice_amplitude", mask1=Anot0_emd, mask2=Anot0_rn)
        if save:
            IO_utils.save([fig0, fig1, fig2, fig3])
        return

    def plot_drude_weight(save=False):
        fig1 = QuantityQuantityPlot("lattice_amplitude", "drude_weight_from_energy_pressure", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd, mask2=Tcutoff_rn)
        fig2 = QuantityQuantityPlot("lattice_amplitude", "drude_weight_from_temperature_entropy", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd, mask2=Tcutoff_rn)
        fig3 = QuantityQuantityPlot("lattice_amplitude", "drude_weight_A0_from_energy_pressure", EMD, RN,
                             quantity_multi_line="temperature", mask1=Tcutoff_emd, mask2=Tcutoff_rn)
        fig1.ax1.set_ylim(ymax=1.6)
        fig2.ax1.set_ylim(ymax=1.6)
        fig3.ax1.set_ylim(ymax=1.6)
        if save:
            IO_utils.save([fig1, fig2, fig3])
        return

    def plot_shear_length(save=False):
        fig1 = QuantityQuantityPlot("lattice_amplitude", "shear_length", EMD, quantity_multi_line="temperature",
                                    mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        fig2 = QuantityQuantityPlot("lattice_amplitude", "shear_length", RN, quantity_multi_line="temperature",
                                    mask1=Tcutoff_rn * Anot0_rn)
        fig1.ax1.set_ylim(ymax=510)
        fig2.ax1.set_ylim(ymax=510)
        fig3 = QuantityQuantityPlot("temperature", "shear_length", EMD, quantity_multi_line="lattice_amplitude",
                                    mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        fig4 = QuantityQuantityPlot("temperature", "shear_length", RN, quantity_multi_line="lattice_amplitude",
                                    mask1=Tcutoff_rn * Anot0_rn)
        fig3.ax1.set_ylim(ymax=510)
        fig4.ax1.set_ylim(ymax=510)
        if save:
            IO_utils.save([fig1, fig2, fig3, fig4])
        return

    # plot_energy_pressure()
    # plot_resistivity()
    # plot_gammaL()
    # plot_conductivities()
    # plot_wf_ratio()
    # plot_EoS()
    # RN = None
    # RN = None
    # plot_sigmaQ()
    # plot_drude_weight()
    plot_shear_length(True)
    # print(np.max(EMD.drude_weight_A0_from_temperature_entropy - EMD.drude_weight_A0_from_energy_pressure))
    # QuantityQuantityPlot("lattice_amplitude", "charge_density", EMD, RN, quantity_multi_line="temperature", mask1=Tcutoff_emd, mask2=Tcutoff_rn)
    # QuantityQuantityPlot("lattice_amplitude", "entropy", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd, mask2=Tcutoff_rn)
    # QuantityQuantityPlot("lattice_amplitude", "charge_density", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd, mask2=Tcutoff_rn)

    # QuantityQuantityPlot("lattice_amplitude", "drudeweight_over_rho", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd, mask2=Tcutoff_rn)
    # QuantityQuantityPlot("lattice_amplitude", "charge_density", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd, mask2=Tcutoff_rn)

    # QuantityQuantityPlot("lattice_amplitude", "alpha_over_sigma", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn*Anot0_rn)
    # QuantityQuantityPlot("lattice_amplitude", "kappabar_over_sigma", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn*Anot0_rn)
    # QuantityQuantityPlot("lattice_amplitude", "kappabar_over_T_sigma", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn*Anot0_rn)

    # QuantityQuantityPlot("lattice_amplitude", "entropy", EMD, RN, quantity_multi_line="temperature", mask1= Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn*Anot0_rn)

    # figlist = [
    #     QuantityQuantityPlot("lattice_amplitude", "conductivity_xx", EMD, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn*Anot0_rn),
    #     QuantityQuantityPlot("lattice_amplitude", "sigmaDC_from_amplitude", EMD, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn),
    #     QuantityQuantityPlot("lattice_amplitude", "sigmaDC_ratio", EMD, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn),
    #     QuantityQuantityPlot("temperature", "sigmaDC_ratio", EMD, RN, quantity_multi_line="lattice_amplitude",
    #                          mask1=Tcutoff_emd*Anot0_emd, mask2=Tcutoff_rn),
    #     QuantityQuantityPlot("lattice_amplitude", "conductivity_xx", EMD, RN, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn),
    #     QuantityQuantityPlot("lattice_amplitude", "sigmaDC_from_amplitude", EMD, RN, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn),
    #     QuantityQuantityPlot("lattice_amplitude", "sigmaDC_ratio", EMD, RN, quantity_multi_line="temperature",
    #                          mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn),
    #     QuantityQuantityPlot("temperature", "sigmaDC_ratio", EMD, RN, quantity_multi_line="lattice_amplitude",
    #                          mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn)
    #
    #     ]
    # IO_utils.save(figlist)
    # d = pd.DataFrame(d={'T': EMD.temperature, 'A': EMD.lattice_amplitude, 'plasmon_frequency_squared': EMD.plasmon_frequency_squared})
    print("plots are build")
    plt.show()
    plt.close()

# TODO:
# clean it up.

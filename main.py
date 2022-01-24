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
import matplotlib
import src.IO_utils as IO_utils
from scipy.optimize import curve_fit
from matplotlib import rcParams
import examples.planckianuniversalityplot as plankianuniversality


plt.style.use(['science','ieee','no-latex'])

# matplotlib.mathtext.SHRINK_FACTOR = 0.2
rcParams['font.family'] = 'DeJavu Sans'
rcParams['font.sans-serif'] = ['Helvetica']

##  Fill in the filenames of the data you want to use. Make sure it is in the data folder.

# EMD_fname = "EMD_T_A_G=0.1000_full.txt"
# EMD_2Dfname = "EMD_T=0.0500_A_G=0.1000.txt"
# EMD_1Dfname = "Unidir_T0.05_G0.1.txt"
RN_fname = "RN_A_T_B0.0000_P0.1000_full.txt"

EMD_fname = "EMD_T_A_G=0.1000_3_zonderAhalf.txt"
# EMD_fname = "EMD_T_A_G=0.1000.txt"

# EMD_fname = EMD_1Dfname

path = "data/"




if __name__ == '__main__':
    plankianuniversality.main()
else:
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

    def func(x, a0, a1):
        return a0 + a1 * x


    def mask_fit(mask, fig, labelprefix, fit_xmax=None):
        x = 1/EMD.lattice_amplitude[mask]
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


    def plot_resistivity(save=False):
        resfig1 = QuantityQuantityPlot("temperature", "resistivity_xx", EMD, RN,
                                       quantity_multi_line="lattice_amplitude",
                                       mask1=Tcutoff_emd, mask2=Tcutoff_rn)
        resfig2 = QuantityQuantityPlot("temperature", "resistivity_xx", EMD,
                                       quantity_multi_line="lattice_amplitude",
                                       mask1=Tcutoff_emd)
        resfig3 = QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD, RN,
                                       quantity_multi_line="temperature",
                                       mask1=Tcutoff_emd, mask2=Tcutoff_rn,)
        resfig4 = QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", EMD,
                                       quantity_multi_line="temperature",
                                       mask1=Tcutoff_emd)
        # T0_02mask = (EMD.temperature < 0.0201) * (EMD.temperature > 0.0199)
        # T0_03mask = (EMD.temperature < 0.0301) * (EMD.temperature > 0.0299)
        # T0_04mask = (EMD.temperature < 0.0401) * (EMD.temperature > 0.0399)
        # T0_05mask = (EMD.temperature < 0.0501) * (EMD.temperature > 0.0499)
        # mask_fit(T0_02mask, resfig4)
        # mask_fit(T0_03mask, resfig4)
        # mask_fit(T0_04mask, resfig4)
        # mask_fit(T0_05mask, resfig4)
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
            IO_utils.save([fig1, fig2, fig3])
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
        fig1 = QuantityQuantityPlot("lattice_amplitude", "one_over_shear_length", EMD, quantity_multi_line="temperature",
                                    mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        fig1.ax1.plot(EMD.lattice_amplitude, EMD.one_over_mu)
        # fig2 = QuantityQuantityPlot("lattice_amplitude", "one_over_shear_length", RN, quantity_multi_line="temperature",
        #                             mask1=Tcutoff_rn * Anot0_rn)
        # fig1.ax1.set_ylim(ymax=510)
        # fig2.ax1.set_ylim(ymax=510)
        fig3 = QuantityQuantityPlot("temperature", "one_over_shear_length", EMD, quantity_multi_line="lattice_amplitude",
                                    mask1=Tcutoff_emd * Anot0_emd, mask2=Tcutoff_rn * Anot0_rn)
        # fig4 = QuantityQuantityPlot("temperature", "one_over_shear_length", RN, quantity_multi_line="lattice_amplitude",
        #                             mask1=Tcutoff_rn * Anot0_rn)
        # fig3.ax1.set_ylim(ymax=510)
        # fig4.ax1.set_ylim(ymax=510)
        if save:
            pass # IO_utils.save([fig1, fig2, fig3, fig4])
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
    # plot_shear_length()
    # print(np.max(EMD.drude_weight_A0_from_temperature_entropy - EMD.drude_weight_A0_from_energy_pressure))
    # print((np.unique(EMD.lattice_amplitude)))
    # print((np.unique(EMD.temperature)))
    custom_mask = np.logical_not((EMD.lattice_amplitude < 0.81) * (EMD.lattice_amplitude > 0.79) * (EMD.temperature < 0.016)
                                 + (EMD.lattice_amplitude < 1.01) * (EMD.lattice_amplitude > 0.99) * (EMD.temperature < 0.008)
                                 )
    # resfig1 = QuantityQuantityPlot("temperature", "resistivity_xx", EMD, quantity_multi_line="lattice_amplitude",
    #                                mask1=Tcutoff_emd*custom_mask, mask2=Tcutoff_rn)
    # resfig1.ax1.set_ylim(ymin=0, ymax=0.6)


    Tmask01= (EMD.temperature > 0.0099) * (EMD.temperature < 0.0101)
    Tmask02 = (EMD.temperature > 0.0199) * (EMD.temperature < 0.0201)
    Tmask03 = (EMD.temperature > 0.0299) * (EMD.temperature < 0.0301)
    Tmask04 = (EMD.temperature > 0.0399) * (EMD.temperature < 0.0401)
    Tmask05 = (EMD.temperature > 0.0499) * (EMD.temperature < 0.0501)
    Tmask06 = (EMD.temperature > 0.0599) * (EMD.temperature < 0.0601)
    Tmask08 = (EMD.temperature > 0.0799) * (EMD.temperature < 0.0801)
    Tmask10 = (EMD.temperature > 0.0999) * (EMD.temperature < 0.1001)

    # Tmask = Tmask01 + Tmask02 + Tmask04 + Tmask06 + Tmask08 + Tmask10
    Tmask = Tmask01 + Tmask03 + Tmask06 + Tmask10
    highAmask = (EMD.lattice_amplitude > 0.02)
    shearlenghtfig = QuantityQuantityPlot("one_over_A", "shear_length", EMD, quantity_multi_line="temperature",
                                          mask1=highAmask * custom_mask * Tmask, linelabels=True, cbar=False)

    high_A_mask = (EMD.lattice_amplitude > .4001) * custom_mask
    mask_fit(Tmask01*high_A_mask, shearlenghtfig, labelprefix="T=0.01, ")
    # mask_fit(Tmask02*high_A_mask, shearlenghtfig, labelprefix="T=0.02, ")
    mask_fit(Tmask03 * high_A_mask, shearlenghtfig, labelprefix="T=0.03, ")
    # mask_fit(Tmask04*high_A_mask, shearlenghtfig, labelprefix="T=0.04, ")
    # mask_fit(Tmask05 * high_A_mask, shearlenghtfig, labelprefix="T=0.05, ")
    mask_fit(Tmask06*high_A_mask, shearlenghtfig, labelprefix="T=0.06, ")
    # mask_fit(Tmask08*high_A_mask, shearlenghtfig, labelprefix="T=0.08, ")
    mask_fit(Tmask10*high_A_mask, shearlenghtfig, labelprefix="T=0.10, ")
    shearlenghtfig.ax1.hlines(np.pi/np.sqrt(2),0,6,"red", "--", label=r"$\pi/(\mu\sqrt{2})$")
    shearlenghtfig.ax1.legend(loc = "upper left")
    shearlenghtfig.ax1.set_xlim(xmax=5)

    sigmaTfig = QuantityQuantityPlot("temperature", "conductivity_T", EMD, quantity_multi_line="lattice_amplitude",
                                     mask1=Anot0_emd*custom_mask*(EMD.temperature>0.0089))
    sigmaTfig.ax1.set_ylim(ymin=0, ymax=10)
    sigmaTfig.ax1.set_xlim(xmin=0, xmax=0.10)
    temp, conductivity_T_lim = utils.sort(EMD.temperature, EMD.conductivity_T_limit)
    sigmaTfig.ax1.plot(temp, conductivity_T_lim, "--", color="red", label=r"$\tau_L = 2 \pi^3\tau_{\hbar}$")
    sigmaTfig.ax1.legend()


    # figlist = [shearlenghtfig, sigmaTfig]
    # IO_utils.save(figlist)
    # d = pd.DataFrame(d={'T': EMD.temperature, 'A': EMD.lattice_amplitude, 'plasmon_frequency_squared': EMD.plasmon_frequency_squared})
    print("plots are build")
    plt.show()
    plt.close()

# TODO:
# clean it up.

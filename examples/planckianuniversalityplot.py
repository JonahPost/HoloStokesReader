# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:01:07 2022

@author: Jonah Post
"""
from src.utils import DataSet
import src.physics as physics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.utils as utils
from src.plot_utils import *
from examples.RN_plots.SamMartijn import plot_RN
from datetime import datetime
import palettable
import os

# rainbow, gist_rainbow, hsv

def main():
    colorschemeBig = mpl.cm.get_cmap('gist_stern', 512)
    cmap1 = mpl.colors.ListedColormap(colorschemeBig(np.linspace(0, 0.5, 256))).reversed()

    colorschemeBig = mpl.cm.get_cmap('Paired', 512)
    cmap2 = mpl.colors.ListedColormap(colorschemeBig(np.linspace(.05, 0.8, 256)))

    cmap = mpl.pylab.cm.inferno
    # Get the colormap colors
    # my_cmap = cmap(np.arange(np.linspace(0, 0.6, cmap.N)))
    # Set alpha
    # my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = cmap(np.arange(0, int(0.6 * cmap.N)))
    my_cmap[:, -1] = np.linspace(.4, 1, int(0.6 * cmap.N))
    # alphas = np.arange(0.7,1.00001,colorschemeBig.N)
    cmap3 = mpl.colors.ListedColormap(my_cmap)

    colors = ["black", "blueviolet", "magenta", "red", "orange"]
    cmap4 = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors, N=5).reversed()

    colors = np.array([mpl.colors.to_rgba("black"), mpl.colors.to_rgba("purple"), mpl.colors.to_rgba("blueviolet"),
                       mpl.colors.to_rgba("magenta"), mpl.colors.to_rgba("red")])
    alphas = np.linspace(.4, 1, 5)
    colors[:, 3] = alphas
    # colors = np.append(colors, [alphas], axis=1)
    print("colors=", colors)

    cmap5 = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors, N=5)

    colors = ["black", "purple", "navy", (0, 0, 1), "red"]
    cmap6 = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors, N=5).reversed()

    cmap7 = mpl.colors.ListedColormap(palettable.scientific.sequential.Imola_6.mpl_colors).reversed()
    cmap7 = mpl.colors.ListedColormap(cmap7(np.linspace(.2, 1, 256)))
    colors = ["black", "purple", "magenta", "red", "darkorange"]
    cmap8 = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors, N=5).reversed()

    colors = ["black", "red", "fuchsia", "dodgerblue", "lime"]
    cmap9 = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors, N=5).reversed()

    cmaps = [cmap1, cmap2, cmap3, cmap4, cmap5, cmap6, cmap7, cmap8]
    cmaps = [cmap3, cmap5]
    for cmap in cmaps:
        make_plot(cmap)

def make_plot(cmap):

    linewidth = 0.6

    path = "data/"

    RN_fname = "RN_ATSeriesHighA.txt"
    RN_A0_fname = "RN_BTSeriesA0"
    EMD_fname = "EMD_T_A_G=0.1000_4.txt"


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
        return xrange, pol(xrange)

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
    figure_size = (4, 3)
    fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2, figsize=figure_size)


    ## Built all relevant EMD masks
    Anot0_emd = (EMD.lattice_amplitude != 0)
    emd_custom_mask = np.logical_not((EMD.lattice_amplitude < 0.81) * (EMD.lattice_amplitude > 0.79) * (EMD.temperature < 0.016)
                                     + (EMD.lattice_amplitude < 0.51) * (EMD.lattice_amplitude > 0.49) * (EMD.temperature < 0.018)
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

    emd_Tmask = emd_Tmask02
    emd_highAmask = (EMD.lattice_amplitude > 0.02)


    emd_shearlenghtfig = QuantityQuantityPlot("one_over_A", "shear_length", EMD, quantity_multi_line="temperature",
                                              mask1=emd_highAmask * emd_custom_mask * emd_Tmask, linelabels=True,
                                              cbar=False, ax=ax3, figure=fig, linecolor="black", linewidth=linewidth, marker=".", cmap=cmap)
    panelC_data = pd.DataFrame(data={"One_over_A": emd_shearlenghtfig.xdata1, "Shearlength": emd_shearlenghtfig.ydata1})
    emd_high_A_mask = (EMD.lattice_amplitude > .4001) * emd_custom_mask
    # mask_fit(emd_Tmask01*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.01, ")
    x_fit, y_fit = mask_fit(emd_Tmask02*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.02, ")
    panelC_fit_data = pd.DataFrame(data={"One_over_A": x_fit, "Shearlength_fit": y_fit})
    # mask_fit(emd_Tmask03 * emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.03, ")
    # mask_fit(emd_Tmask04*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.04, ")
    # mask_fit(emd_Tmask05 * emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.05, ")
    # mask_fit(emd_Tmask06*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.06, ")
    # mask_fit(emd_Tmask08*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.08, ")
    # mask_fit(emd_Tmask10*emd_high_A_mask, emd_shearlenghtfig, labelprefix="T=0.10, ")
    emd_shearlenghtfig.ax1.hlines(np.pi/np.sqrt(2),0,6,"red", "--", linewidth=0.5, label=r"$\pi/(\mu\sqrt{2})$")
    emd_shearlenghtfig.ax1.set_xlim(xmax=5)
    emd_shearlenghtfig.ax1.annotate("C", xy=(0.95, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')

    emd_Amask04 = (EMD.lattice_amplitude > 0.399) * (EMD.lattice_amplitude < 0.401)
    emd_Amask05 = (EMD.lattice_amplitude > 0.499) * (EMD.lattice_amplitude < 0.501)
    emd_Amask08 = (EMD.lattice_amplitude > 0.799) * (EMD.lattice_amplitude < 0.801)
    emd_Amask10 = (EMD.lattice_amplitude > 0.999) * (EMD.lattice_amplitude < 1.001)
    emd_Amask12 = (EMD.lattice_amplitude > 1.199) * (EMD.lattice_amplitude < 1.201)
    emd_Amask15 = (EMD.lattice_amplitude > 1.499) * (EMD.lattice_amplitude < 1.501)
    emd_Amask16 = (EMD.lattice_amplitude > 1.599) * (EMD.lattice_amplitude < 1.601)
    emd_Amask20 = (EMD.lattice_amplitude > 1.999) * (EMD.lattice_amplitude < 2.001)
    emd_Amask = emd_Amask04 + emd_Amask08 + emd_Amask12 + emd_Amask16 + emd_Amask20


    emd_sigmaTfig = QuantityQuantityPlot("temperature", "conductivity_T", EMD, quantity_multi_line="lattice_amplitude",
                                         mask1=emd_Amask*emd_custom_mask*(EMD.temperature>0.0089), ax=ax1, figure=fig,
                                         cbar=False, linelabels=True, linewidth=1, marker="", cmap=cmap)
    data_mask = emd_custom_mask*(EMD.temperature>0.0089)
    print(len(EMD.conductivity_T[data_mask*emd_Amask04]), len(np.append(np.zeros(7).fill(np.nan),EMD.conductivity_T[data_mask*emd_Amask08])),len(EMD.conductivity_T[data_mask*emd_Amask12]),len(EMD.conductivity_T[data_mask*emd_Amask16]),len(EMD.conductivity_T[data_mask*emd_Amask20]))
    panelA_data = pd.DataFrame(data={'Temperature': EMD.temperature[data_mask*emd_Amask04],
                                     'SigmaT_A04': EMD.conductivity_T[data_mask*emd_Amask04],
                                     'SigmaT_A08': np.append(np.empty(7),EMD.conductivity_T[data_mask*emd_Amask08]),
                                     'SigmaT_A12': EMD.conductivity_T[data_mask*emd_Amask12],
                                     'SigmaT_A16': EMD.conductivity_T[data_mask*emd_Amask16],
                                     'SigmaT_A20': EMD.conductivity_T[data_mask*emd_Amask20]})
    emd_sigmaTfig.ax1.set_ylim(ymin=0, ymax=10)
    emd_sigmaTfig.ax1.set_xlim(xmin=0, xmax=0.10)
    emd_temp, emd_conductivity_T_lim = utils.sort(EMD.temperature, EMD.conductivity_T_limit)
    emd_sigmaTfig.ax1.plot(emd_temp, emd_conductivity_T_lim, "--", linewidth=0.5, color="red", label=r"$\tau_L = 2 \pi^3\tau_{\hbar}$")
    panelA_saturation_data = pd.DataFrame(data={"Temperature":EMD.temperature, "SigmaT_saturation": EMD.conductivity_T_limit})
    emd_sigmaTfig.ax1.annotate("A", xy=(0.95, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')

    plot_RN(ax2, ax4, cmap)
    rn_high_A_mask = (EMD.lattice_amplitude > .4001)


    ax1.set_xlim(xmin=0, xmax=0.06)
    ax2.set_xlim(xmin=0, xmax=0.06)
    ax3.set_xlim(xmin=0, xmax=3)
    ax4.set_xlim(xmin=0, xmax=3)

    ax2.set_ylim(ymin=0, ymax=65)
    ax3.set_ylim(ymin=0, ymax=10)
    ax4.set_ylim(ymin=0, ymax=10)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper left")

    fig.tight_layout()


    ## save the dataframes
    panelA_data.to_csv('export_data/panelA_data.csv', sep='\t')
    panelA_saturation_data.to_csv('export_data/panelA_saturation_data.csv', sep='\t')
    panelC_data.to_csv('export_data/panelC_data.csv', sep='\t')
    panelC_fit_data.to_csv('export_data/panelC_fit_data.csv', sep='\t')

    save = False
    # save = True
    if save:
        folder_name = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        os.mkdir(folder_name)
        print("new folder made")
        fig.savefig(folder_name + "/PlanckianUniversality.png")
        print("plots are saved")

    plt.show()
    plt.close()

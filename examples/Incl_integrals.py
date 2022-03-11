import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import pandas as pd
import src.physics as physics



def main():
    RN_path = "data/data_incl_integrals/"
    RNA_fname = "FlorisRN-A_with_integrals.txt"
    RNA = DataSet(model="RN", fname=RN_path + RNA_fname, snellius=True) ## AT series at G=0.2
    RNB_fname = "FlorisRN-B_with_integrals.txt"
    RNB = DataSet(model="RN", fname=RN_path + RNB_fname, snellius=True) ## AG series at T=0.02
    RNC_fname = "FlorisRN-C_with_integrals.txt"
    RNC = DataSet(model="RN", fname=RN_path + RNC_fname, snellius=True) ## GT series for some A's
    GR_path = "data/GR_data/"
    GRA_fname = "FlorisGR-A.txt"
    GRA = DataSet(model="GR", fname=GR_path + GRA_fname, snellius=True)
    GRB_fname = "FlorisGR-B.txt"
    GRB = DataSet(model="GR", fname=GR_path + GRB_fname, snellius=True)
    def make_integral_properties(dataset):
        dataset.homogeneous_energy = (2 + .5 * (dataset.chem_pot ** 2)) / (dataset.chem_pot ** 3)
        dataset.homogeneous_pressure = dataset.homogeneous_energy / 2
        dataset.homogeneous_entropy = 4 * np.pi / (dataset.chem_pot ** 2)
        dataset.homogeneous_charge_density = 1 / dataset.chem_pot
        dataset.s_over_rho_homogeneous = 4 * np.pi / dataset.chem_pot
        dataset.hom_constants = dataset.homogeneous_entropy/(4*np.pi*(dataset.homogeneous_energy+dataset.homogeneous_pressure))
        dataset.IntegralExpB0 = dataset.data["IntegralExpB0"].to_numpy()
        dataset.IntegralExpB0AtOverQtt = dataset.data["IntegralExpB0AtOverQtt"].to_numpy()
        dataset.s_over_rho = dataset.entropy / dataset.charge_density
        dataset.s_over_rho_pheno = 4 * np.pi * dataset.IntegralExpB0 / dataset.IntegralExpB0AtOverQtt
        dataset.s_over_rho_ratio = dataset.s_over_rho / dataset.s_over_rho_pheno
        dataset.sigmaQ0 = 1 / dataset.IntegralExpB0
        dataset.X = ((4 * np.pi) ** 2) * dataset.temperature / (dataset.sigmaQ0 * dataset.kappabar_xx)
        dataset.sqrt_one_over_X = np.sqrt(1/dataset.X)
        dataset.X_over_T = dataset.X / dataset.temperature
        dataset.XT = dataset.X * dataset.temperature
        dataset.GammaPheno1 = dataset.X * dataset.sigmaQ0
        dataset.sqrt_one_over_GammaPheno1 = np.sqrt(1 / dataset.GammaPheno1)
        dataset.GammaL_from_kappabar = (dataset.entropy ** 2) * dataset.temperature / (dataset.kappabar_xx * (dataset.energy + dataset.pressure))
        dataset.sqrt_one_over_GammaL_from_kappabar = np.sqrt(1/dataset.GammaL_from_kappabar)
        dataset.Gamma_ratio = dataset.GammaL_from_kappabar / dataset.X
        dataset.shear_length1 = np.sqrt(dataset.homogeneous_entropy/(4*np.pi*(dataset.homogeneous_energy+dataset.homogeneous_pressure)*dataset.GammaPheno1))
        dataset.shear_length2 = np.sqrt(dataset.homogeneous_entropy/(4*np.pi*(dataset.homogeneous_energy+dataset.homogeneous_pressure)*dataset.GammaL_from_kappabar))
        dataset.WFratio = dataset.kappabar_xx / (dataset.conductivity_xx * dataset.temperature)
    def mask_fit(mask, dataset, ax, labelprefix=None, fit_xmax=None):
        x = 1 / dataset.lattice_amplitude[mask]
        y = dataset.sqrt_one_over_GammaL_from_kappabar[mask]
        x, y = utils.remove_nan(x, y)
        x, y = utils.sort(x, y)
        popt, pol = utils.pol_fit(x, y, type="linear")
        xmax = x[0]
        if fit_xmax:
            xmax = fit_xmax
        xrange = np.linspace(0, xmax)
        ax.plot(xrange, pol(xrange), linestyle=(0, (1, 1)), c="k", linewidth=.5)
        # print(r"${:.2g}+{:.2g}x$".format(*popt))
        return xrange, pol(xrange)
    def plot_X():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaL_from_kappabar", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaPheno1", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "Gamma_ratio", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".")
        axs[1, 1].legend()
        print(RNB.Gamma_ratio[(RNB.lattice_amplitude<.3)])
        axs[0, 0].set_ylabel(r"$\sqrt{1/\Gamma_L}$")
        axs[1, 0].set_ylabel(r"$\sqrt{1/X}$")
        axs[1, 1].set_ylabel(r"$\Gamma_L / X$")
        axs[0, 0].set_xlim(xmax=3)
        axs[1, 0].set_xlim(xmax=3)
        axs[1, 1].set_xlim(xmax=3)
        axs[0, 0].set_ylim(ymax=20)
        axs[1, 0].set_ylim(ymax=6)
        axs[0, 0].set_title(r"RN-2D, T=0.02")
        fig.tight_layout()
        fig, axs = plt.subplots(2, 2)
        Tmask = (RNA.temperature > 0.0099)*(RNA.temperature < 0.0101) + (RNA.temperature == 0.05) + (RNA.temperature == 0.1)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaL_from_kappabar", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaPheno1", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "Gamma_ratio", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=Tmask, marker=".")
        axs[1, 1].legend()
        axs[0, 0].set_ylabel(r"$\sqrt{1/\Gamma_L}$")
        axs[1, 0].set_ylabel(r"$\sqrt{1/X}$")
        axs[1, 1].set_ylabel(r"$\Gamma_L / X$")
        axs[0, 0].set_xlim(xmax=3)
        axs[1, 0].set_xlim(xmax=3)
        axs[1, 1].set_xlim(xmax=3)
        axs[0, 0].set_ylim(ymax=20)
        axs[1, 0].set_ylim(ymax=4)
        axs[0, 0].set_title(r"RN-2D, G=0.2")
        fig.tight_layout()

        fig, ax = plt.subplots(1, 1)
        QuantityQuantityPlot("temperature", "XT", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=ax, figure=fig, marker=".")
        ax.legend()
        # ax.set_ylabel(r"$X/T$")
        # ax.set_xlim(xmax=4)
        # ax.set_ylim(ymax=40)
        ax.set_title(r"RN-2D, T=0.02")

    def plot_GammaL_from_kappabar():
        fig, ax = plt.subplots(1, 1)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaL_from_kappabar", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=ax, figure=fig, marker=".")
        ax.legend()
        ax.set_ylabel(r"$\sqrt{1/\Gamma_{pheno,2}}$")
        ax.set_xlim(xmax=1)
        ax.set_ylim(ymax=10)
        ax.set_title(r"RN-2D, T=0.02")
        lowA = (RNB.lattice_amplitude > 2)
        maskG01 = (RNB.periodicity == 0.10) * lowA
        maskG02 = (RNB.periodicity == 0.20) * lowA
        maskG04 = (RNB.periodicity == 0.40) * lowA
        maskG06 = (RNB.periodicity == 0.60) * lowA
        maskG10 = (RNB.periodicity == 1) * lowA
        for period in np.unique(RNB.periodicity)[:-2]:
            print(period)
            mask_fit((RNB.periodicity == period) * lowA, RNB, ax)
        mask_fit((RNB.periodicity == 0.60) * (RNB.lattice_amplitude > 5), RNB, ax)
        # mask_fit((RNB.periodicity == 1) * (RNB.lattice_amplitude > 6), RNB, ax)
        fig.tight_layout()
    def plot_s_over_rho():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("lattice_amplitude", "s_over_rho", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
        QuantityQuantityPlot("lattice_amplitude", "s_over_rho_pheno", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
        QuantityQuantityPlot("lattice_amplitude", "s_over_rho_ratio", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
        axs[1, 0].set_ylabel(r"$\left(\frac{s}{\rho}\right)_{pheno}$")
        axs[1, 1].set_ylabel(r"$\frac{s}{\rho} / \left(\frac{s}{\rho}\right)_{pheno}$")
        axs[0, 1].legend(axs[0, 0].get_legend_handles_labels()[0], axs[0, 0].get_legend_handles_labels()[1])
        axs[0, 0].set_title("RN-2D, T=0.02")
        fig.tight_layout()

    def plot_GR_conductivities():
        fig, axs = plt.subplots(2, 2)
        GRA.conductivity_T = GRA.conductivity_xx * GRA.temperature
        GRA.alpha_T = GRA.alpha_xx * GRA.temperature
        GRA.kappabar_Tsquared = GRA.kappabar_xx * GRA.temperature**2
        QuantityQuantityPlot("temperature", "conductivity_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
        QuantityQuantityPlot("temperature", "alpha_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
        QuantityQuantityPlot("temperature", "kappabar_Tsquared", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
        fig.tight_layout()

    make_integral_properties(RNA)
    make_integral_properties(RNB)
    make_integral_properties(RNC)
    plot_X()
    # plot_GammaL_from_kappabar()
    # plot_s_over_rho()
    # plot_GR_conductivities()
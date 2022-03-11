import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import pandas as pd
import src.physics as physics

def main():
    RN_path = "data/RN_data/"
    RNA_fname = "FlorisRN-A.txt"
    RNA = DataSet(model="RN", fname=RN_path + RNA_fname, snellius=True)
    RNB_fname = "FlorisRN-B.txt"
    RNB = DataSet(model="RN", fname=RN_path + RNB_fname, snellius=True)
    RNC_fname = "FlorisRN-C.txt"


    GR_path = "data/GR_data/"
    GRA_fname = "FlorisGR-A.txt"
    GRA = DataSet(model="GR", fname=GR_path + GRA_fname, snellius=True)
    GRB_fname = "FlorisGR-B.txt"
    GRB = DataSet(model="GR", fname=GR_path + GRB_fname, snellius=True)
    # GRC_fname = "FlorisGR-C.txt"
    # GRC = DataSet(model="GR", fname=GR_path + GRC_fname, snellius=True)

    # print(np.unique(RNA.periodicity), np.unique(RNA.temperature), np.unique(RNA.lattice_amplitude))
    # print(np.unique(GRA.periodicity), np.unique(GRA.temperature), np.unique(GRA.lattice_amplitude))
    # print(np.unique(RNB.periodicity), np.unique(RNB.temperature), np.unique(RNB.lattice_amplitude))
    # print(np.unique(GRB.periodicity), np.unique(GRB.temperature), np.unique(GRB.lattice_amplitude))
    # print(np.unique(RNC.periodicity))

    def plot_conductivities():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "conductivity_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
        QuantityQuantityPlot("temperature", "alpha_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
        QuantityQuantityPlot("temperature", "kappabar_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
        # axs[0, 0].legend()
        # axs[0, 1].legend()
        # axs[1, 0].legend()
        axs[1, 1].legend(axs[1, 0].get_legend_handles_labels()[0], axs[1, 0].get_legend_handles_labels()[1])
        axs[0, 0].set_ylim(ymax=65)
        axs[0, 1].set_ylim(ymax=65)
        axs[1, 0].set_ylim(ymax=65)
        axs[0, 0].set_title("RN, G=0.2")
        fig.tight_layout()

    def plot_integrals():
        Tmask = (RNA.temperature > 0.0099)*(RNA.temperature < 0.0101) + (RNA.temperature == 0.05) + (RNA.temperature == 0.1)
        RNA.one_over_IntegralExpB0 = 1 / RNA.IntegralExpB0
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "IntegralExpB0", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Tmask, marker=".")
        Amask = (RNA.lattice_amplitude == 0.6)
        QuantityQuantityPlot("one_over_A", "one_over_IntegralExpB0", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Tmask, marker=".")
        RNA.sigmaQ0 = RNA.conductivity_xx - RNA.temperature * (RNA.alpha_xx * RNA.alphabar_xx) / RNA.kappabar_xx
        QuantityQuantityPlot("one_over_A", "sigmaQ0", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=Tmask, marker=".")
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[0, 0].set_ylabel(r"$\int{e^{B^{(0)}}}$")
        axs[1, 0].set_ylabel(r"$\sigma_{(Q=0)} = 1/\int{e^{B^{(0)}}}$")
        axs[1, 1].set_ylabel(r"$\sigma_{Q=0} = \sigma - T\frac{\alpha \bar{\alpha}}{\bar{\kappa}}$")
        axs[0, 0].set_ylim(ymax=1.1)
        axs[1, 0].set_ylim(ymax=2.5)
        axs[1, 1].set_ylim(ymax=2.5)
        fig.tight_layout()
    def plot_GammaPheno():
        Tmask = (RNA.temperature > 0.0099)*(RNA.temperature < 0.0101) + (RNA.temperature == 0.05) + (RNA.temperature == 0.1)
        RNA.sigmaQ0 = 1 / RNA.IntegralExpB0
        RNA.X = ((4*np.pi)**2)*RNA.temperature/(RNA.sigmaQ0*RNA.kappabar_xx)
        RNA.GammaPheno = RNA.X * RNA.sigmaQ
        RNA.GammaPheno_over_T = RNA.GammaPheno / RNA.temperature
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "sigmaQ0", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "X", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "GammaPheno", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "GammaPheno_over_T", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=Tmask, marker=".")
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[0, 0].set_ylim(ymax=1.1)
        axs[1, 1].set_ylim(ymax=20)
        axs[0, 0].set_ylabel(r"$\sigma_{(Q=0)} = 1/\int{e^{B^{(0)}}}$")
        axs[0, 1].set_ylabel(r"$X = \frac{(4 \pi)^2 T}{\sigma_{(Q=0)} \bar{\kappa}}$")
        axs[1, 0].set_ylabel(r"$\Gamma_{pheno} = X  \sigma_{(Q=0)} $")
        axs[1, 1].set_ylabel(r"$\Gamma_{pheno} /T$")
        fig.tight_layout()

        fig, axs = plt.subplots(2, 2)
        RNA.X_over_A2 = RNA.X / (RNA.lattice_amplitude**2)
        RNA.one_over_X = 1 / RNA.X
        QuantityQuantityPlot("lattice_amplitude", "X", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("lattice_amplitude", "X_over_A2", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "one_over_X", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Tmask, marker=".")
        # QuantityQuantityPlot("one_over_A", "GammaPheno_over_T", RNA, quantity_multi_line="temperature",
        #                      cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=Tmask, marker=".")
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[0, 0].set_ylabel(r"$X$")
        axs[0, 1].set_ylabel(r"$X/A^2$")
        axs[1, 0].set_ylabel(r"$1/X$")
        fig.tight_layout()

        RNA.GammaPheno_2 = 1/( RNA.kappabar_xx *(RNA.energy + RNA.pressure) / ((RNA.entropy**2)*(RNA.temperature) ))
        RNA.GammaPheno_2_over_T = RNA.GammaPheno_2 / RNA.temperature
        RNA.GammaPheno_ratio = RNA.GammaPheno_2 / RNA.GammaPheno
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "GammaPheno", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Tmask, marker=".")
        # QuantityQuantityPlot("one_over_A", "GammaPheno_over_T", RNA, quantity_multi_line="temperature",
        #                      cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "GammaPheno_2", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Tmask, marker=".")
        QuantityQuantityPlot("one_over_A", "GammaPheno_ratio", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=Tmask, marker=".")
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        # axs[1, 1].legend()
        axs[0, 0].set_ylabel(r"$\Gamma_{pheno,1} = X  \sigma_{(Q=0)} $")
        # axs[1, 1].set_ylim(ymax=2)
        # axs[0, 1].set_ylabel(r"$\Gamma_{pheno,1} /T$")
        axs[1, 0].set_ylabel(r"$\Gamma_{pheno,2} = \frac{s^2 T}{\bar{\kappa}(\mathcal{E}+\mathcal{P})}$")
        axs[1, 1].set_ylabel(r"$\Gamma_{pheno,2} / \Gamma_{pheno,1}$")
        fig.tight_layout()

        RNA.sqrt_one_over_X = np.sqrt(RNA.one_over_X)
        RNA.sqrt_one_over_GammaPheno2 = np.sqrt(1 / RNA.GammaPheno_2)
        fig, ax = plt.subplots(1, 1)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_X", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=ax, figure=fig, mask1=Tmask, marker=".")
        ax.legend()
        ax.set_ylabel(r"$\sqrt{1/X}$")
        fig.tight_layout()
        fig, ax = plt.subplots(1, 1)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaPheno2", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=ax, figure=fig, mask1=Tmask, marker=".")
        ax.legend()
        ax.set_ylabel(r"$\sqrt{1/\Gamma_{pheno,2}}$")
        fig.tight_layout()

    # plot_conductivities()
    RNA_fname = "FlorisRN-A_with_integrals.txt"
    RNA = DataSet(model="RN", fname=RN_path + RNA_fname, snellius=True)
    RNA.IntegralExpB0 = RNA.data["IntegralExpB0"].to_numpy()
    # plot_integrals()
    plot_GammaPheno()
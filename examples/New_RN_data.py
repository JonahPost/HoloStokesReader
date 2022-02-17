import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import src.physics as physics

def main():
    path = "data/BData/"
    RN_fname = "FlorisRN-B.txt"
    RN = DataSet(model="RN", fname=path + RN_fname, snellius=True)
    physics.calc_properties(RN)
    print(RN.temperature)
    # fig, axs = plt.subplots(2, 2)
    # QuantityQuantityPlot("lattice_amplitude", "energy", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "pressure", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 1], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "energy_pressure_ratio", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "energy_plus_pressure", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 1], figure=fig)
    # fig.tight_layout()

    Anot0 = (RN.lattice_amplitude > 0.001)
    Acutoff = (RN.lattice_amplitude >2)
    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("lattice_amplitude", "conductivity_xx", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 1], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("lattice_amplitude", "alpha_xx", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
                         ax=axs[1, 0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("lattice_amplitude", "kappabar_xx", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
                         ax=axs[1, 1], figure=fig, mask1=Anot0)
    # axs[0, 0].set_ylim(ymax=20)
    axs[0, 1].set_ylim(ymax=20)
    axs[1, 0].set_ylim(ymax=20)
    axs[1, 1].set_ylim(ymax=20)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    fig.tight_layout()
    #
    # fig, axs = plt.subplots(1, 2)
    # QuantityQuantityPlot("lattice_amplitude", "entropy", RN, quantity_multi_line="periodicity", cbar=False,
    #                      ax=axs[0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "entropy_over_T", RN, quantity_multi_line="periodicity", cbar=False,
    #                      ax=axs[1], figure=fig)
    # fig.tight_layout()

    fig, axs = plt.subplots(2, 2)
    Alow = (RN.lattice_amplitude < 0.501)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 0], figure=fig, mask1=Anot0*Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 1], figure=fig, mask1=Anot0*Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[1, 0], figure=fig, mask1=Anot0*Alow)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[0, 0].set_xlim(xmin=0, xmax=0.5)
    axs[0, 1].set_xlim(xmin=0, xmax=0.5)
    axs[1, 0].set_xlim(xmin=0, xmax=0.5)
    fig.tight_layout()

    fig, axs = plt.subplots(2,1)
    QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_alpha", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
                         ax=axs[0], figure=fig, mask1=Anot0*Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_kappabar", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
                         ax=axs[1], figure=fig, mask1=Anot0*Alow)
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlim(xmin=0, xmax=0.5)
    axs[1].set_xlim(xmin=0, xmax=0.5)
    axs[0].set_ylim(ymax=1.1)
    axs[1].set_ylim(ymax=1.1)
    fig.tight_layout()


    fig, axs = plt.subplots(2, 2)
    Alow = (RN.lattice_amplitude < 10)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 0], figure=fig, mask1=Anot0 * Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0, 1], figure=fig, mask1=Anot0 * Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[1, 0], figure=fig, mask1=Anot0 * Alow)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[0, 0].set_xlim(xmin=0, xmax=9)
    axs[0, 1].set_xlim(xmin=0, xmax=9)
    axs[1, 0].set_xlim(xmin=0, xmax=9)
    fig.tight_layout()

    fig, axs = plt.subplots(2, 1)
    QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_alpha", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0], figure=fig, mask1=Anot0 * Alow)
    QuantityQuantityPlot("lattice_amplitude", "gamma_reldiff_sigma_kappabar", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[1], figure=fig, mask1=Anot0 * Alow)
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlim(xmin=0, xmax=9)
    axs[1].set_xlim(xmin=0, xmax=9)
    # axs[0].set_ylim(ymax=1.1)
    # axs[1].set_ylim(ymax=1.1)
    fig.tight_layout()

    fig, axs = plt.subplots(2, 1)
    QuantityQuantityPlot("lattice_amplitude", "sigmaQ_from_sigma_alpha", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("lattice_amplitude", "sigmaQ_from_sigma_kappabar", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True,
                         ax=axs[1], figure=fig, mask1=Anot0)
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlim(xmin=0, xmax=9)
    axs[1].set_xlim(xmin=0, xmax=9)
    fig.tight_layout()
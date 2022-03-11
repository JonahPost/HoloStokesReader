import numpy as np
from src.utils import DataSet
from src.plot_utils import *
import src.physics as physics

def main():
    path = "data/RN_data/BData/"
    RN_fname = "FlorisRN-B.txt"
    RN = DataSet(model="RN", fname=path + RN_fname, snellius=True)
    physics.calc_properties(RN)
    Anot0 = (RN.lattice_amplitude > 0.001)
    Acutoff = (RN.lattice_amplitude >2)
    Alow = (RN.lattice_amplitude < 0.501)
    RN.homogeneous_energy = (2 + .5 * (RN.chem_pot**2))/(RN.chem_pot**3)
    RN.homogeneous_pressure = RN.homogeneous_energy/2
    RN.homogeneous_entropy = 4 * np.pi / (RN.chem_pot**2)
    RN.homogeneous_charge_density = 1 / RN.chem_pot
    RN.shear_length = np.sqrt(RN.homogeneous_entropy/(4*np.pi*(RN.homogeneous_energy+RN.homogeneous_pressure)*RN.gamma_L_from_sigma))


    # ## THERMODYNAMIC QUANTITIES
    # fig, axs = plt.subplots(2, 2)
    # QuantityQuantityPlot("lattice_amplitude", "energy", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "pressure", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 1], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "energy_pressure_ratio", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "pressure_ratio", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 1], figure=fig)
    # fig.tight_layout()
    # fig, axs = plt.subplots(2, 2)
    # QuantityQuantityPlot("lattice_amplitude", "equation_of_state", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "equation_of_state_ratio", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[0, 1], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "entropy", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 0], figure=fig)
    # QuantityQuantityPlot("lattice_amplitude", "charge_density", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=axs[1, 1], figure=fig)
    # fig.tight_layout()

    # ## CONDUCTIVITIES
    # fig, axs = plt.subplots(2, 2)
    # QuantityQuantityPlot("lattice_amplitude", "resistivity_xx", RN, quantity_multi_line="periodicity",
    #                      cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Anot0)
    # QuantityQuantityPlot("lattice_amplitude", "conductivity_xx", RN, quantity_multi_line="periodicity",
    #                      cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Anot0)
    # QuantityQuantityPlot("lattice_amplitude", "alpha_xx", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
    #                      ax=axs[1, 0], figure=fig, mask1=Anot0)
    # QuantityQuantityPlot("lattice_amplitude", "kappabar_xx", RN, quantity_multi_line="periodicity", cbar=False, linelabels=True,
    #                      ax=axs[1, 1], figure=fig, mask1=Anot0)
    # axs[0, 1].set_ylim(ymax=20)
    # axs[1, 0].set_ylim(ymax=20)
    # axs[1, 1].set_ylim(ymax=20)
    # axs[0, 0].legend()
    # axs[0, 1].legend()
    # axs[1, 0].legend()
    # axs[1, 1].legend()
    # fig.tight_layout()

    def plot_gammas():
        # GAMMA_L - ALow
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Anot0*Alow)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Anot0*Alow)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Anot0*Alow)
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[0, 0].set_xlim(xmin=0, xmax=0.5)
        axs[0, 1].set_xlim(xmin=0, xmax=0.5)
        axs[1, 0].set_xlim(xmin=0, xmax=0.5)
        fig.tight_layout()

        # GAMMA_L RATIOS - Alow
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

        # GAMMA_L
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_sigma", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Anot0)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_alpha", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Anot0)
        QuantityQuantityPlot("lattice_amplitude", "gamma_L_from_kappabar", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Anot0)
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[0, 0].set_xlim(xmin=0, xmax=9)
        axs[0, 1].set_xlim(xmin=0, xmax=9)
        axs[1, 0].set_xlim(xmin=0, xmax=9)
        fig.tight_layout()

        ## GAMMA_L RATIOS
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("lattice_amplitude", "gamma_ratio_sigma_alpha", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=Anot0)
        QuantityQuantityPlot("lattice_amplitude", "gamma_ratio_sigma_kappabar", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=Anot0)
        QuantityQuantityPlot("lattice_amplitude", "gamma_ratio_alpha_kappabar", RN, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=Anot0)
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].set_yticks([])
        axs[1, 1].set_xticks([])
        axs[0, 0].set_xlim(xmin=0, xmax=9)
        axs[0, 1].set_xlim(xmin=0, xmax=9)
        axs[1, 0].set_xlim(xmin=0, xmax=9)
        axs[1, 1].text(.5, .5, r"$\sigma_{Q} = 0$", fontsize=10, ha='center', va='center')
        axs[0, 0].set_ylim(ymin=0, ymax=2)
        axs[0, 1].set_ylim(ymin=0, ymax=2)
        axs[1, 0].set_ylim(ymin=0, ymax=2)
        axs[1, 1].text(.5,.5, r"$\sigma_{Q,(\sigma,\alpha)} = \frac{\sigma - \frac{\rho}{s}\alpha}{1+\frac{\mu \rho}{sT}}$", fontsize=10,ha='center', va='center')
        axs[0, 0].set_ylim(ymin=0, ymax=1.1)
        axs[0, 1].set_ylim(ymin=-2, ymax=2)
        axs[1, 0].set_ylim(ymin=-2, ymax=2)
        axs[1, 1].text(.5,.5, r"$\sigma_{Q,(\sigma,\bar{\kappa})} = \frac{\sigma - \frac{\rho^2}{s^2T}\bar{\kappa}}{1-\frac{\mu^2 \rho^2}{s^2T^2}}$", fontsize=10,ha='center', va='center')
        axs[0, 0].set_ylim(ymin=-10, ymax=10)
        axs[0, 1].set_ylim(ymin=0, ymax=1.1)
        axs[1, 0].set_ylim(ymin=-1, ymax=1.1)
        axs[1, 1].text(.5,.5, r"$\sigma_{Q,(\alpha,\bar{\kappa})} = \frac{\frac{\rho}{sT}\bar{\kappa} - \alpha}{\frac{\mu^2 \rho}{T^2s}+\frac{\mu}{T}}$", fontsize=10,ha='center', va='center')
        axs[0, 0].set_ylim(ymin=0, ymax=10)
        axs[0, 1].set_ylim(ymin=0, ymax=10)
        axs[1, 0].set_ylim(ymin=0, ymax=1.1)
        fig.tight_layout()
        return

    def mask_fit(mask, ax, labelprefix=None, fit_xmax=None):
        x = 1 / RN.lattice_amplitude[mask]
        y = RN.shear_length[mask]
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
        return xrange, pol(xrange)

    hydro_mask = (RN.periodicity == 0.10) * (RN.lattice_amplitude < .8) + \
                 (RN.periodicity == 0.20) * (RN.lattice_amplitude < 1) + \
                 (RN.periodicity == 0.40) * (RN.lattice_amplitude < 1.3) + \
                 (RN.periodicity == 0.60) * (RN.lattice_amplitude < 1.9) + \
                 (RN.periodicity == 1) * (RN.lattice_amplitude < 3)

    ## ALPHA PLOTS
    print(RN.s_over_rho[0])
    RN.s_over_rho_homogeneous = 4*np.pi/RN.chem_pot
    # RN.s_over_rho = RN.s_over_rho_homogeneous
    RN.alpha_fit = RN.drude_weight_from_energy_pressure * RN.s_over_rho * (1 / RN.gamma_L_from_sigma) # - (RN.sigmaQ_from_sigma_kappabar / RN.temperature)
    RN.alpha_fitratio = RN.alpha_fit / RN.alpha_xx
    fig, ax = plt.subplots(1, 1)
    QuantityQuantityPlot("lattice_amplitude", "alpha_fit", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linestyle="--")
    QuantityQuantityPlot("lattice_amplitude", "alpha_xx", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linelabels=True, marker=".")
    ax.legend()
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(xmin=0, xmax=9)
    ax.set_ylim(ymax=50)
    fig, ax = plt.subplots(1, 1)
    QuantityQuantityPlot("lattice_amplitude", "alpha_fitratio", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linestyle="--", linewidth=.5)
    QuantityQuantityPlot("lattice_amplitude", "alpha_fitratio", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linelabels=True, mask1=None, marker=".")
    ax.legend()
    ax.set_ylabel(r"$\frac{\alpha_{fit}}{\alpha_{data}}$")
    ax.set_xlim(xmin=0, xmax=9)
    ax.set_ylim(ymin=0,ymax=2)
    #
    # # ## KAPPABAR PLOTS
    # # RN.kappabar_fit = RN.drude_weight_from_energy_pressure * (RN.s_over_rho)**2 * RN.temperature * (1 / RN.gamma_L_from_sigma) + (RN.sigmaQ_from_sigma_alpha / RN.temperature)
    # # RN.kappabar_fitratio = RN.kappabar_fit / RN.kappabar_xx
    # # fig, ax = plt.subplots(1, 1)
    # # QuantityQuantityPlot("lattice_amplitude", "kappabar_fit", RN, quantity_multi_line="periodicity",
    # #                      cbar=False, ax=ax, figure=fig, linestyle="--")
    # # QuantityQuantityPlot("lattice_amplitude", "kappabar_xx", RN, quantity_multi_line="periodicity",
    # #                      cbar=False, ax=ax, figure=fig, linelabels=True)
    # # ax.legend()
    # # ax.set_ylabel(r"$\bar{\kappa}$")
    # # ax.set_xlim(xmin=0, xmax=9)
    # # ax.set_ylim(ymax=50)
    # # fig, ax = plt.subplots(1, 1)
    # # QuantityQuantityPlot("lattice_amplitude", "kappabar_fitratio", RN, quantity_multi_line="periodicity",
    # #                      cbar=False, ax=ax, figure=fig, linelabels=True, linewidth=.5)
    # # # QuantityQuantityPlot("lattice_amplitude", "kappabar_fitratio", RN, quantity_multi_line="periodicity",
    # # #                      cbar=False, ax=ax, figure=fig, linelabels=True, mask1=hydro_mask)
    # # ax.legend()
    # # ax.set_ylabel(r"$\frac{\bar{\kappa}_{fit}}{\bar{\kappa}_{data}}$")
    # # ax.set_xlim(xmin=0, xmax=9)
    # # ax.set_ylim(ymin=0,ymax=2)
    #
    # ## CONDUCTIVITY
    #
    # fig, ax = plt.subplots(1, 1)
    # QuantityQuantityPlot("one_over_A", "conductivity_xx", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=ax, figure=fig, linelabels=True, linestyle="-", marker=".")
    # ax.legend()
    # ax.set_xlim(xmin=0, xmax=1)
    # ax.set_ylim(ymin=0,ymax=20)

    # ## SHEAR LENGTH
    # fig, ax = plt.subplots(1, 1)
    # QuantityQuantityPlot("one_over_A", "shear_length", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=ax, figure=fig, linelabels=True, linestyle="-", marker=".")
    # ax.hlines(np.pi / np.sqrt(2), 0, 6, "red", "--", linewidth=0.5, label=r"$\pi/(\mu\sqrt{2})$")
    # # fit_mask = hydro_mask
    # # mask_fit(fit_mask * (RN.periodicity == 0.10), ax, labelprefix="G=0.10, ")
    # # mask_fit(fit_mask * (RN.periodicity == 0.20), ax, labelprefix="G=0.20, ")
    # # mask_fit(fit_mask * (RN.periodicity == 0.40), ax, labelprefix="G=0.40, ")
    # # mask_fit(fit_mask * (RN.periodicity == 0.60), ax, labelprefix="G=0.60, ")
    # # mask_fit(fit_mask * (RN.periodicity == 01.0), ax, labelprefix="G=1.00, ")
    #
    # ax.legend()
    # ax.set_xlim(xmin=0, xmax=3)
    # ax.set_ylim(ymin=0,ymax=10)
    #
    # fig, ax = plt.subplots(1, 1)
    # QuantityQuantityPlot("lattice_amplitude", "stress_energy_trace", RN, quantity_multi_line="periodicity",
    #                      cbar=False, ax=ax, figure=fig, linelabels=True)

    ####################################################################################################################

    RN.s_over_rho_eff = RN.kappabar_xx / (RN.alpha_xx * RN.temperature)
    RN.s_over_rho_ratio = RN.s_over_rho_eff / RN.s_over_rho
    RN.drudeness = RN.rho_over_s * RN.alpha_xx / (RN.conductivity_xx)
    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("lattice_amplitude", "s_over_rho", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
    QuantityQuantityPlot("lattice_amplitude", "s_over_rho_eff", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
    QuantityQuantityPlot("lattice_amplitude", "s_over_rho_ratio", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
    QuantityQuantityPlot("lattice_amplitude", "drudeness", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[0, 0].set_ylim(ymax=20)
    axs[0, 1].set_ylim(ymax=20)
    axs[0, 1].set_ylabel(r"$\left(\frac{s}{\rho}\right)_{eff} = \frac{\bar{\kappa}}{\alpha T}$")
    axs[1, 0].set_ylabel(r"$\left(\frac{s}{\rho}\right)_{eff} / \frac{s}{\rho} = \frac{\bar{\kappa}}{\alpha T}\frac{\rho}{s}$")
    axs[1, 1].set_ylabel(r"$\frac{\alpha}{\sigma}\frac{\rho}{s}$")
    fig.tight_layout()

    RN.sigmaQ_math = RN.conductivity_xx - RN.alpha_xx * RN.rho_over_s
    RN.sigmaQ_math_eff = RN.conductivity_xx - RN.alpha_xx / RN.s_over_rho_eff
    RN.sigmaQ0 = RN.conductivity_xx - RN.temperature*(RN.alpha_xx*RN.alphabar_xx)/RN.kappabar_xx

    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("lattice_amplitude", "sigmaQ_math", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
    QuantityQuantityPlot("lattice_amplitude", "sigmaQ_math_eff", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
    QuantityQuantityPlot("lattice_amplitude", "sigmaQ0", RN, quantity_multi_line="periodicity",
                         cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[0, 0].set_ylim(ymax=3)
    axs[0, 1].set_ylim(ymax=3)
    axs[1, 0].set_ylim(ymax=3)
    axs[0, 0].set_ylabel(r"$\sigma - \alpha \left(\frac{s}{\rho}\right)_{thermo}$")
    axs[0, 1].set_ylabel(r"$\sigma - \alpha \left(\frac{s}{\rho}\right)_{eff}$")
    axs[1, 0].set_ylabel(r"$\sigma_{Q=0} = \sigma - T\frac{\alpha \bar{\alpha}}{\bar{\kappa}}$")
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    QuantityQuantityPlot("one_over_A", "kappabar_xx", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linelabels=True)
    ax.set_ylim(ymax=10)
    ax.set_xlim(xmax=2)
    ax.legend()

    fig, ax = plt.subplots(1, 1)
    QuantityQuantityPlot("lattice_amplitude", "kappa_xx", RN, quantity_multi_line="periodicity",
                         cbar=False, ax=ax, figure=fig, linelabels=True)
    ax.legend()
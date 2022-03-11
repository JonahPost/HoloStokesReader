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
    # oldGR = DataSet(model="GR", fname="data/EMD_T_A_G=0.1000_full.txt")
    GR_path = "data/GR_data/"
    GRA_fname = "GR-A.txt"
    GRA = DataSet(model="GR", fname=GR_path + GRA_fname, snellius=True)
    GRB_fname = "FlorisGR-B.txt"
    GRB = DataSet(model="GR", fname=GR_path + GRB_fname, snellius=True)
    GRC_fname = "GR-C.txt"
    GRC = DataSet(model="GR", fname=GR_path + GRC_fname, snellius=True)
    for datamodel in [RNA, RNB, RNC, GRA, GRB, GRC]:
        datamodel.conductivity_T = datamodel.conductivity_xx * datamodel.temperature
        datamodel.resistivity_over_T = datamodel.resistivity_xx / datamodel.temperature
        datamodel.sigmaQ0 = datamodel.conductivity_xx - datamodel.alpha_xx**2 * datamodel.temperature / datamodel.kappabar_xx
        datamodel.sigmaQ0_T = datamodel.sigmaQ0*datamodel.temperature
        datamodel.alpha_T = datamodel.alpha_xx * datamodel.temperature
        datamodel.kappabar_over_T = datamodel.kappabar_xx / (datamodel.temperature)
        datamodel.kappabar_over_Tsquared = datamodel.kappabar_xx / (datamodel.temperature**2)
        datamodel.kappa = datamodel.kappabar_xx - datamodel.alpha_xx**2 * datamodel.temperature / datamodel.conductivity_xx
        datamodel.kappa_over_T = datamodel.kappa / datamodel.temperature
        datamodel.kappa_over_T2 = datamodel.kappa / datamodel.temperature**2
        datamodel.homogeneous_energy = (2 + .5 * (datamodel.chem_pot ** 2)) / (datamodel.chem_pot ** 3)
        datamodel.homogeneous_pressure = datamodel.homogeneous_energy / 2
        datamodel.homogeneous_entropy = 4 * np.pi / (datamodel.chem_pot ** 2)
        datamodel.homogeneous_charge_density = 1 / datamodel.chem_pot
        datamodel.energy_plus_pressure = datamodel.energy + datamodel.pressure
        datamodel.energy_over_pressure = datamodel.energy / datamodel.pressure
        datamodel.stress_energy_trace = (-datamodel.energy + datamodel.pressure_x + datamodel.pressure_y)
        datamodel.homogeneous_energy_plus_pressure = (datamodel.homogeneous_energy + datamodel.homogeneous_pressure)
        datamodel.homogeneous_first_law_ratio = (datamodel.homogeneous_energy_plus_pressure) / (datamodel.homogeneous_entropy*datamodel.temperature + datamodel.homogeneous_charge_density)
        datamodel.first_law_ratio = (datamodel.energy + datamodel.pressure) / (datamodel.entropy*datamodel.temperature + datamodel.charge_density)
        datamodel.s_over_T = datamodel.entropy / datamodel.temperature
        datamodel.s_over_rho = datamodel.entropy / datamodel.charge_density
        datamodel.s_over_rho_homogeneous = 4*np.pi/datamodel.chem_pot
        datamodel.sigmaQ_from_sigma_kappabar = datamodel.conductivity_xx - ((datamodel.charge_density/datamodel.entropy)**2)*(datamodel.kappabar_xx/datamodel.temperature)
        datamodel.drude_weight = (datamodel.charge_density ** 2) / (datamodel.energy + datamodel.pressure)
        datamodel.Gamma_L_from_sigma = datamodel.drude_weight / (datamodel.conductivity_xx-datamodel.sigmaQ_from_sigma_kappabar)
        datamodel.Gamma_L_from_alpha = (datamodel.entropy / datamodel.charge_density) * datamodel.drude_weight / (datamodel.alpha_xx)
        datamodel.Gamma_L_from_kappabar = (datamodel.entropy / datamodel.charge_density)**2 * datamodel.drude_weight / (datamodel.kappabar_xx / datamodel.temperature)
        datamodel.sqrt_one_over_GammaL_from_kappabar = np.sqrt(1/datamodel.Gamma_L_from_kappabar)
        datamodel.shear_length = np.sqrt(datamodel.homogeneous_entropy/(4*np.pi*(datamodel.homogeneous_energy+datamodel.homogeneous_pressure)*datamodel.Gamma_L_from_kappabar))
        datamodel.alpha_fit = datamodel.drude_weight*(datamodel.entropy/datamodel.charge_density) *(1/datamodel.Gamma_L_from_sigma)
        datamodel.alpha_ratio = datamodel.alpha_fit/datamodel.alpha_xx
        datamodel.alpha_error = np.abs( (datamodel.alpha_ratio-1) * datamodel.alpha_xx )
        # datamodel.Gamma_L_from_alpha_error = ((datamodel.entropy*datamodel.charge_density)/(datamodel.energy+datamodel.pressure))*(1/datamodel.alpha_xx**2)*datamodel.alpha_error
        ## Calculate the Lorentz factor for the Wiederman-Franz law
        datamodel.s2_over_rho2 = datamodel.s_over_rho ** 2
        datamodel.Lbar = datamodel.kappabar_xx / (datamodel.conductivity_xx * datamodel.temperature)
        datamodel.Lbar_over_T2 = datamodel.Lbar / datamodel.temperature**2
        datamodel.Lbar_comparison = datamodel.Lbar / datamodel.s2_over_rho2
        datamodel.L = datamodel.kappa / (datamodel.conductivity_xx * datamodel.temperature)
        datamodel.L_over_T2 = datamodel.L / datamodel.temperature ** 2
        datamodel.L_comparison = datamodel.L / datamodel.s2_over_rho2
        ## Deduce integrals from the conductivities
        datamodel.X_over_IntegralExpB0 = (4*np.pi)**2 * datamodel.temperature / datamodel.kappabar_xx
        datamodel.X_over_IntegralExpB0_over_T = datamodel.X_over_IntegralExpB0 / datamodel.temperature
        datamodel.sqrt_IntegralExpB0_over_X = np.sqrt(1/datamodel.X_over_IntegralExpB0)
        datamodel.Gamma_ratio = datamodel.Gamma_L_from_kappabar / datamodel.X_over_IntegralExpB0
        try: ## Extract the integraldata if it exists
            datamodel.IntegralExpB0 = datamodel.data["IntegralExpB0"].to_numpy()
            datamodel.X_normalized = 0.03424658 * ((4 * np.pi) ** 2) * datamodel.temperature * datamodel.IntegralExpB0 / (datamodel.kappabar_xx)
            datamodel.Gamma_L_error = np.abs(datamodel.Gamma_L_from_kappabar - datamodel.X_normalized)
            datamodel.shear_length_error = .5 * (datamodel.shear_length / datamodel.X_normalized) * datamodel.Gamma_L_error

        except:
            pass
    def plot_fig1():
        fig, axs = plt.subplots(2, 2)
        AmaskGR = (GRA.lattice_amplitude==0.4)+(GRA.lattice_amplitude==1.2)+(GRA.lattice_amplitude==2)+ (GRA.lattice_amplitude==2.8)+(GRA.lattice_amplitude==3.6)
        QuantityQuantityPlot("temperature", "conductivity_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".", mask1=(GRA.lattice_amplitude>0.2))
        QuantityQuantityPlot("temperature", "conductivity_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".", mask1=(RNA.lattice_amplitude>0.2))
        QuantityQuantityPlot("one_over_A", "shear_length", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".")
        # QuantityQuantityPlot("one_over_A", "shear_length", GRB, quantity_multi_line="periodicity",
        #                      cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".")
        axs[0, 0].legend(ncol=2)
        axs[0, 1].legend(ncol=2)
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[0, 0].set_xlim(xmax=0.05)
        axs[0, 0].set_ylim(ymax=10)
        axs[0, 1].set_xlim(xmax=0.05)
        axs[0, 1].set_ylim(ymax=70)
        axs[1, 0].set_xlim(xmax=3)
        axs[1, 0].set_ylim(ymax=10)
        axs[1, 1].set_xlim(xmax=3)
        axs[1, 1].set_ylim(ymax=10)
        axs[0, 0].annotate("A", xy=(0.94, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')
        axs[0, 1].annotate("B", xy=(0.94, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')
        axs[1, 0].annotate("C", xy=(0.94, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')
        axs[1, 1].annotate("D", xy=(0.94, 1.02), xycoords="axes fraction", fontsize=7, fontweight='bold')
        fig.tight_layout()


        # for G in np.unique(RNB.periodicity):
        #     Gmask = (RNB.periodicity==G)
        #     fig, ax = plt.subplots(1, 1)
        #     QuantityQuantityPlot("one_over_A", "shear_length", RNB, quantity_multi_line="periodicity",
        #                          cbar=False, linelabels=True, ax=ax, figure=fig, marker=".", yerr="shear_length_error",
        #                          mask1=Gmask, cbarmin=0.1, cbarmax=1)
        #     ax.legend()
        #     ax.set_title(r"RN-2D, $T=0.02$")
        #     ax.set_xlim(xmax=3)
        #     ax.set_ylim(ymax=10)
        #     fig.tight_layout()
    def alpha_fits():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("lattice_amplitude", "alpha_fit", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=False, ax=axs[0,0], figure=fig, linestyle="--")
        QuantityQuantityPlot("lattice_amplitude", "alpha_xx", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0,0], figure=fig)
        QuantityQuantityPlot("lattice_amplitude", "alpha_ratio", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
        QuantityQuantityPlot("temperature", "alpha_fit", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=False, ax=axs[1,0], figure=fig, linestyle="--")
        QuantityQuantityPlot("temperature", "alpha_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1,0], figure=fig)
        QuantityQuantityPlot("temperature", "alpha_ratio", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[0, 0].set_title(r"RN-2D, $T=0.02$")
        axs[0, 1].set_title(r"RN-2D, $T=0.02$")
        axs[1, 0].set_title(r"RN-2D, $G=0.2$")
        axs[1, 1].set_title(r"RN-2D, $G=0.2$")
        axs[0, 0].set_ylim(ymax=50)
        axs[0, 0].set_xlim(xmax=9)
        axs[0, 1].set_xlim(xmax=9)
        axs[1, 0].set_ylim(ymax=200)
        axs[1, 0].set_xlim(xmax=0.1)
        axs[1, 1].set_xlim(xmax=0.1)
        fig.tight_layout()
    def plot_thermodynamics():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "energy_plus_pressure", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".")
        QuantityQuantityPlot("temperature", "s_over_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".")
        QuantityQuantityPlot("temperature", "charge_density", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".")
        QuantityQuantityPlot("temperature", "first_law_ratio", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".")
        axs[0, 1].legend()
        axs[1, 1].set_ylabel(r"$\frac{\mathcal{E}+\mathcal{P}}{Ts+\mu\rho}$")
        fig.tight_layout()
    def plot_conductivities():
        fig, axs = plt.subplots(3, 2, figsize=(4, 4))
        print(np.unique(GRA.periodicity), np.unique(RNA.periodicity))
        QuantityQuantityPlot("temperature", "conductivity_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".", mask1=(GRA.lattice_amplitude>0.1)*(GRA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        QuantityQuantityPlot("temperature", "alpha_xx", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".", mask1=(GRA.lattice_amplitude>0.1)*(GRA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        QuantityQuantityPlot("temperature", "kappabar_over_Tsquared", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[2, 0], figure=fig, marker=".", mask1=(GRA.lattice_amplitude>0.1)*(GRA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        QuantityQuantityPlot("temperature", "conductivity_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".", mask1=(RNA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        QuantityQuantityPlot("temperature", "alpha_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".", mask1=(RNA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        QuantityQuantityPlot("temperature", "kappabar_over_T", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[2, 1], figure=fig, marker=".", mask1=(RNA.lattice_amplitude<4.5), cbarmin =0.4, cbarmax=4.4)
        axs[0, 0].set_title(r"GR-2D, $G=0.2$")
        axs[0, 1].set_title(r"RN-2D, $G=0.2$")
        axs[0, 1].set_ylabel(r"$\sigma$")
        axs[1, 0].set_ylabel(r"$\alpha$")
        axs[1, 1].set_ylabel(r"$\alpha$")
        axs[2, 0].set_ylabel(r"$\bar{\kappa}/T^2$")
        axs[2, 1].set_ylabel(r"$\bar{\kappa}/T$")
        for axrow in axs:
            for ax in axrow:
                ax.legend(ncol=2)
                ax.set_xlim(xmax=0.05)
        axs[0, 0].set_ylim(ymax=40)
        axs[0, 1].set_ylim(ymax=40)
        axs[1, 0].set_ylim(ymax=250)
        axs[1, 1].set_ylim(ymax=250)
        axs[2, 0].set_ylim(ymax=17000)
        axs[2, 1].set_ylim(ymax=1000)
        fig.tight_layout()
    def plot_kappabar_kappa():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "kappabar_over_Tsquared", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
        QuantityQuantityPlot("temperature", "kappabar_over_T", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
        QuantityQuantityPlot("temperature", "kappa_over_T2", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
        QuantityQuantityPlot("temperature", "kappa_over_T", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
        for axrow in axs:
            for ax in axrow:
                ax.legend(ncol=2)
        axs[0, 0].set_ylim(ymax=17000)
        axs[0, 1].set_ylim(ymax=1000)
        fig.tight_layout()
    def plot_sigma_sigmaQ0():
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "conductivity_xx", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig)
        QuantityQuantityPlot("temperature", "conductivity_xx", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig)
        QuantityQuantityPlot("temperature", "sigmaQ0", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig)
        QuantityQuantityPlot("temperature", "sigmaQ0", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig)
        for axrow in axs:
            for ax in axrow:
                ax.legend(ncol=1)
        fig.tight_layout()
    def plot_Lbar(): ## Wiedermand Franz ratio
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "Lbar_over_T2", GRB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "Lbar", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "Lbar_comparison", GRB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "Lbar_comparison", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".")
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=1)
                ax.set_xlim(xmax=1)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        axs[1, 0].legend(ncol=1)
        axs[1, 1].legend(ncol=1)
        axs[0, 0].set_ylabel(r"$\frac{\bar{L}}{T^2}=\frac{\bar{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"$\bar{L}=\frac{\bar{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $T=0.02$")
        axs[0, 1].set_title(r"RN-2D, $T=0.02$")
        fig.tight_layout()

        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "Lbar_over_T2", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".", cbarmin=0.4, cbarmax=8, mask1=(GRA.lattice_amplitude>0.2))
        QuantityQuantityPlot("temperature", "Lbar", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".", cbarmin=0.4, cbarmax=8)
        QuantityQuantityPlot("temperature", "Lbar_comparison", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".", cbarmin=0.4, cbarmax=8, mask1=(GRA.lattice_amplitude>0.2))
        QuantityQuantityPlot("temperature", "Lbar_comparison", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".", cbarmin=0.4, cbarmax=8)
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=2)
                ax.set_xlim(xmax=0.05)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        axs[0, 0].set_ylabel(r"$\frac{\bar{L}}{T^2}=\frac{\bar{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"$\bar{L}=\frac{\bar{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $G=0.2$")
        axs[0, 1].set_title(r"RN-2D, $G=0.2$")
        fig.tight_layout()

        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "Lbar_over_T2", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "Lbar", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "Lbar_comparison", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "Lbar_comparison", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 4), multi_line_around=1)
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=2)
                ax.set_xlim(xmax=0.05)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        # axs[0, 0].set_ylim(ymax=5)
        axs[0, 1].set_xlim(xmax=0.02)
        axs[1, 1].set_xlim(xmax=0.02)
        axs[0, 0].set_ylabel(r"$\frac{\bar{L}}{T^2}=\frac{\bar{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"$\bar{L}=\frac{\bar{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"$\bar{L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $A=4$")
        axs[0, 1].set_title(r"RN-2D, $A=4$")
        fig.tight_layout()
    def plot_L(): ## Wiedermand Franz ratio
        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("one_over_A", "L_over_T2", GRB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "L", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "L_comparison", GRB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".")
        QuantityQuantityPlot("one_over_A", "L_comparison", RNB, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".")
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=1)
                ax.set_xlim(xmax=1)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        axs[1, 0].legend(ncol=1)
        axs[1, 1].legend(ncol=1)
        axs[0, 0].set_ylabel(r"$\frac{{L}}{T^2}=\frac{{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"${L}=\frac{\bar{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $T=0.02$")
        axs[0, 1].set_title(r"RN-2D, $T=0.02$")
        fig.tight_layout()

        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "L_over_T2", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".", cbarmin=0.4, cbarmax=8, mask1=(GRA.lattice_amplitude>0.2))
        QuantityQuantityPlot("temperature", "L", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".", cbarmin=0.4, cbarmax=8)
        QuantityQuantityPlot("temperature", "L_comparison", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".", cbarmin=0.4, cbarmax=8, mask1=(GRA.lattice_amplitude>0.2))
        QuantityQuantityPlot("temperature", "L_comparison", RNA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".", cbarmin=0.4, cbarmax=8)
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=2)
                ax.set_xlim(xmax=0.05)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        axs[0, 0].set_ylabel(r"$\frac{{L}}{T^2}=\frac{{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"${L}=\frac{{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $G=0.2$")
        axs[0, 1].set_title(r"RN-2D, $G=0.2$")
        fig.tight_layout()

        fig, axs = plt.subplots(2, 2)
        QuantityQuantityPlot("temperature", "L_over_T2", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "L", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "L_comparison", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), multi_line_around=1)
        QuantityQuantityPlot("temperature", "L_comparison", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 4), multi_line_around=1)
        for axrow in axs:
            for ax in axrow:
                # ax.legend(ncol=2)
                ax.set_xlim(xmax=0.05)
        axs[0, 0].legend(ncol=1)
        axs[0, 1].legend(ncol=2)
        # axs[0, 0].set_ylim(ymax=5)
        axs[0, 1].set_xlim(xmax=0.02)
        axs[1, 1].set_xlim(xmax=0.02)
        axs[0, 0].set_ylabel(r"$\frac{{L}}{T^2}=\frac{{\kappa}}{\sigma T^3}$")
        axs[0, 1].set_ylabel(r"${L}=\frac{{\kappa}}{\sigma T}$")
        axs[1, 0].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[1, 1].set_ylabel(r"${L}/\frac{s^2}{\rho^2}$")
        axs[0, 0].set_title(r"GR-2D, $A=4$")
        axs[0, 1].set_title(r"RN-2D, $A=4$")
        fig.tight_layout()

    def plot_resistivity():
        fig, axs = plt.subplots(3, 2, figsize=(4,5))
        QuantityQuantityPlot("temperature", "resistivity_xx", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 0.02), logx=True, logy=True, multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_xx", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 0.02), logx=True, logy=True, multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_xx", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 1), logx=True, logy=True, multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_xx", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 1), logx=True, logy=True, multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_xx", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[2, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), logx=True, logy=True, multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_xx", RNC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[2, 1], figure=fig, marker=".",
                             mask1=(RNC.lattice_amplitude == 4), logx=True, logy=True, multi_line_around=1)

        axs[0, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='lower right')
        axs[1, 0].legend(loc='lower right')
        axs[1, 1].legend(loc='lower right')
        axs[2, 0].legend(loc='lower right')
        axs[2, 1].legend(loc='lower right')
        fig.tight_layout()
        axs[0, 0].set_xlim(xmin=0.01, xmax=0.1)
        axs[1, 0].set_xlim(xmin=0.01, xmax=0.1)
        axs[2, 0].set_xlim(xmin=0.01, xmax=0.1)
        axs[0, 1].set_xlim(xmin=0.0007 ,xmax=0.012)
        axs[1, 1].set_xlim(xmin=0.0007 ,xmax=0.012)
        axs[2, 1].set_xlim(xmin=0.0007 ,xmax=0.012)
        axs[0, 0].set_title(r"GR-2D, A=0.02")
        axs[0, 1].set_title(r"RN-2D, A=0.02")
        axs[1, 0].set_title(r"GR-2D, A=1")
        axs[1, 1].set_title(r"RN-2D, A=1")
        axs[2, 0].set_title(r"GR-2D, A=4")
        axs[2, 1].set_title(r"RN-2D, A=4")
        fig.tight_layout()
        fig, axs = plt.subplots(3, 2, figsize=(4,5))
        QuantityQuantityPlot("temperature", "resistivity_over_T", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 0.02), multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_over_T", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 1), multi_line_around=1)
        QuantityQuantityPlot("temperature", "resistivity_over_T", GRC, quantity_multi_line="periodicity",
                             cbar=False, linelabels=True, ax=axs[2, 0], figure=fig, marker=".",
                             mask1=(GRC.lattice_amplitude == 4), multi_line_around=1)
        axs[0, 0].legend()
        axs[0, 0].set_title(r"GR-2D, $A=0.02$")
        axs[1, 0].set_title(r"GR-2D, $A=1$")
        axs[2, 0].set_title(r"GR-2D, $A=4$")

        fig.tight_layout()
        fig, axs = plt.subplots(1, 1)
        QuantityQuantityPlot("temperature", "resistivity_over_T", GRA, quantity_multi_line="lattice_amplitude",
                             cbar=False, linelabels=True, ax=axs, figure=fig, marker=".")
        axs.legend()
        axs.set_title(r"GR-2D, $G=0.2$")
        fig.tight_layout()

    def plot_Gammaratio():
        fig, axs = plt.subplots(3, 2, figsize=(4, 5))
        TmaskGR = (GRA.temperature > 0.0099) * (GRA.temperature < 0.0101) + (GRA.temperature > 0.039)*(GRA.temperature < 0.041) + (GRA.temperature > 0.099)*(GRA.temperature < 0.101)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaL_from_kappabar", GRA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 0], figure=fig, mask1=TmaskGR, marker=".")
        QuantityQuantityPlot("one_over_A", "sqrt_IntegralExpB0_over_X", GRA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 0], figure=fig, mask1=TmaskGR, marker=".")
        QuantityQuantityPlot("one_over_A", "Gamma_ratio", GRA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[2, 0], figure=fig, mask1=TmaskGR, marker=".")

        TmaskRN = (RNA.temperature > 0.0099) * (RNA.temperature < 0.0101) + (RNA.temperature > 0.039)*(RNA.temperature < 0.041) + (RNA.temperature > 0.099)*(RNA.temperature < 0.101)
        QuantityQuantityPlot("one_over_A", "sqrt_one_over_GammaL_from_kappabar", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0, 1], figure=fig, mask1=TmaskRN, marker=".")
        QuantityQuantityPlot("one_over_A", "sqrt_IntegralExpB0_over_X", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1, 1], figure=fig, mask1=TmaskRN, marker=".")
        QuantityQuantityPlot("one_over_A", "Gamma_ratio", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[2, 1], figure=fig, mask1=TmaskRN, marker=".")
        axs[0, 0].set_ylabel(r"$\sqrt{1/\Gamma_L}$")
        axs[1, 0].set_ylabel(r"$\sqrt{\int e^{B^{(0)}}/X}$")
        axs[2, 0].set_ylabel(r"$\Gamma_L / \frac{X}{\int e^{B^{(0)}}}$")
        axs[0, 1].set_ylabel(r"$\sqrt{1/\Gamma_L}$")
        axs[1, 1].set_ylabel(r"$\sqrt{\int e^{B^{(0)}}/X}$")
        axs[2, 1].set_ylabel(r"$\Gamma_L / \frac{X}{\int e^{B^{(0)}}}$")
        for axrow in axs:
            for ax in axrow:
                ax.legend()
                ax.set_xlim(xmax=3)
        axs[0, 0].set_ylim(ymax=30)
        axs[0, 1].set_ylim(ymax=30)
        axs[1, 0].set_ylim(ymax=6)
        axs[1, 1].set_ylim(ymax=6)
        axs[0, 0].set_title(r"GR-2D, G=0.2")
        axs[0, 1].set_title(r"RN-2D, G=0.2")
        fig.tight_layout()
        fig, axs = plt.subplots(1, 2, figsize=(4,2))
        TmaskGR = (GRA.temperature > 0.0099) * (GRA.temperature < 0.0101) + (GRA.temperature > 0.039) * (
                    GRA.temperature < 0.041) + (GRA.temperature > 0.099) * (GRA.temperature < 0.101)
        QuantityQuantityPlot("one_over_A", "X_over_IntegralExpB0_over_T", GRA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[0], figure=fig, mask1=TmaskGR, marker=".")
        TmaskRN = (RNA.temperature > 0.0099) * (RNA.temperature < 0.0101) + (RNA.temperature > 0.039) * (
                    RNA.temperature < 0.041) + (RNA.temperature > 0.099) * (RNA.temperature < 0.101)
        QuantityQuantityPlot("one_over_A", "X_over_IntegralExpB0_over_T", RNA, quantity_multi_line="temperature",
                             cbar=False, linelabels=True, ax=axs[1], figure=fig, mask1=TmaskRN, marker=".")
        axs[0].set_title(r"GR-2D, G=0.2")
        axs[1].set_title(r"RN-2D, G=0.2")
        for ax in axs:
            ax.legend()
            ax.set_ylabel(r"$\frac{1}{T}\frac{X}{e^{B^{(0)}}}$")
        fig.tight_layout()

    plot_thermodynamics()
    # plot_resistivity()
    # plot_conductivities()
    # plot_kappabar_kappa()
    # plot_sigma_sigmaQ0()
    # plot_Lbar()
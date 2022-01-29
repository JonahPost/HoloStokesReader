import numpy as np
from src.plot_utils import *
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','no-latex'])

SMALL_SIZE = 4
MEDIUM_SIZE = 4
BIGGER_SIZE = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)



def plot_energy_pressure(datamodel):
    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("temperature", "energy", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0, 0], figure=fig)
    QuantityQuantityPlot("temperature", "pressure", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0, 1], figure=fig)
    QuantityQuantityPlot("temperature", "energy_pressure_ratio", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1, 0], figure=fig)
    QuantityQuantityPlot("temperature", "energy_plus_pressure", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1, 1], figure=fig)

    fig.tight_layout()

def plot_conductivities(datamodel):
    Anot0 = (datamodel.lattice_amplitude > 0.01)
    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("temperature", "resistivity_xx", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0, 0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("temperature", "conductivity_xx", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0, 1], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("temperature", "alpha_xx", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1, 0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("temperature", "kappabar_xx", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1, 1], figure=fig, mask1=Anot0)

    fig.tight_layout()

def plot_entropy(datamodel):
    fig, axs = plt.subplots(1, 2)
    QuantityQuantityPlot("temperature", "entropy", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0], figure=fig)
    QuantityQuantityPlot("temperature", "entropy_over_T", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1], figure=fig)

    fig.tight_layout()

def plot_drude_weight(datamodel):
    Tcutoff = (datamodel.temperature > 0.0199)
    fig, axs = plt.subplots(2, 2)
    # QuantityQuantityPlot("lattice_amplitude", "drude_weight_from_energy_pressure", datamodel, quantity_multi_line="temperature", cbar=False,
    #                      ax=axs[0, 0], figure=fig, mask1=Tcutoff)
    # axs[0,0].set_ylim(ymax=0.03)
    # QuantityQuantityPlot("lattice_amplitude", "drude_weight_from_temperature_entropy", datamodel, quantity_multi_line="temperature", cbar=False,
    #                      ax=axs[0, 1], figure=fig, mask1=Tcutoff)
    # QuantityQuantityPlot("lattice_amplitude", "drude_weight_A0", datamodel, quantity_multi_line="temperature", cbar=False,
    #                      ax=axs[1, 0], figure=fig, mask1=Tcutoff)

    QuantityQuantityPlot("temperature", "drude_weight_from_energy_pressure", datamodel,
                         quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0, 0], figure=fig, mask1=Tcutoff)
    QuantityQuantityPlot("temperature", "drude_weight_from_energy_pressure", datamodel,
                         quantity_multi_line="lattice_amplitude", cbar=True,
                         ax=axs[0, 1], figure=fig, mask1=Tcutoff)
    axs[0, 1].set_ylim(ymax=0.035)
    QuantityQuantityPlot("temperature", "drude_weight_from_temperature_entropy", datamodel,
                         quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[1, 0], figure=fig, mask1=Tcutoff)
    QuantityQuantityPlot("temperature", "drude_weight_A0", datamodel, quantity_multi_line="lattice_amplitude",
                         cbar=False,
                         ax=axs[1, 1], figure=fig, mask1=Tcutoff)

    fig.tight_layout()

def plot_universality(datamodel):
    Anot0 = (datamodel.lattice_amplitude > 0.01)
    Tcutoff = (datamodel.temperature > 0.0199)
    fig, axs = plt.subplots(2, 2)
    QuantityQuantityPlot("temperature", "conductivity_T", datamodel, quantity_multi_line="lattice_amplitude", cbar=False,
                         ax=axs[0,0], figure=fig, mask1=Anot0)
    QuantityQuantityPlot("one_over_A", "shear_length", datamodel, quantity_multi_line="temperature", cbar=False,
                         ax=axs[1,0], figure=fig, mask1=Anot0*Tcutoff)
    QuantityQuantityPlot("one_over_A", "shear_length_alt1", datamodel, quantity_multi_line="temperature", cbar=False,
                         ax=axs[1, 1], figure=fig, mask1=Anot0*Tcutoff)
    # QuantityQuantityPlot("one_over_A", "shear_length_alt2", datamodel, quantity_multi_line="temperature",
    #                      cbar=False,
    #                      ax=axs[1, 2], figure=fig, mask1=Anot0*Tcutoff)
    axs[1, 0].set_ylim(ymin=0, ymax=10)
    axs[1, 0].text(.95, .05, r"$\sqrt{\frac{s_h}{4\pi}\frac{\sigma_{xx}}{\rho_h^2}}$",
                              horizontalalignment='right',
                              verticalalignment='bottom',
                              transform=axs[1, 0].transAxes)
    axs[1, 1].set_ylim(ymin=0, ymax=10)
    axs[1, 1].text(.95, .05, r"$\sqrt{\frac{s}{4\pi}\frac{\sigma_{xx}}{\rho^2}}$",
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   transform=axs[1, 1].transAxes)
    # axs[1, 2].set_ylim(ymin=0, ymax=10)
    # axs[1, 2].text(0.01, .95, r"$\sqrt{\frac{s_h}{4\pi (\mathcal{E} + \mathcal{P})}\frac{\sigma_{xx}}{\omega^2_{p,h}}}$",
    #                horizontalalignment='left',
    #                verticalalignment='top',
    #                transform=axs[1, 2].transAxes)
    fig.tight_layout()

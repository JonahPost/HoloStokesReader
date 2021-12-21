# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:02:00 2021

@author: Jonah Post
"""
import numpy as np

def calc_properties(model):
    model.s_over_rho = model.entropy / model.charge_density
    # compute_s_over_rho_A0(model)
    model.rho_over_s = 1/model.s_over_rho

    model.kappa_xx = model.kappabar_xx - model.alpha_xx ** 2 * model.temperature / model.conductivity_xx
    model.kappabar_over_T = model.kappabar_xx / model.temperature

    ## Relative conductivities
    model.alpha_over_sigma = model.alpha_xx / model.conductivity_xx
    model.kappabar_over_sigma = model.kappabar_xx / model.conductivity_xx
    model.kappabar_over_T_sigma = model.kappabar_over_sigma / model.temperature

    ## Drude Weight / omega_p squared
    model.drude_weight_from_energy_pressure = \
        (model.charge_density ** 2 / (model.energy + model.pressure))
    model.drude_weight_from_temperature_entropy = \
        (model.charge_density ** 2 / (model.entropy * model.temperature + model.charge_density))
    # Also for A=0, A-independent weight
    compute_drude_weight_A0(model)

    model.drudeweight_over_rho = model.drude_weight_from_temperature_entropy / model.charge_density

    ## Thermodynamics
    model.equation_of_state = model.energy + model.pressure - model.temperature * model.entropy - model.charge_density
    model.energy_pressure_ratio = model.energy / model.pressure
    model.one_over_mu = 1 / model.chem_pot
    model.resistivity_xx = 1 / model.conductivity_xx
    model.sigmaDC_from_amplitude = np.sqrt(3) * ( 1 + model.lattice_amplitude**2 )**2 / ( 2*np.pi*(model.lattice_amplitude**2)*np.sqrt(4 + 6*(model.lattice_amplitude**2)) * model.temperature)
    model.sigmaDC_ratio = model.conductivity_xx/model.sigmaDC_from_amplitude
    ## Wiedermann-Franz ratio
    model.s2_over_rho2 = model.s_over_rho**2
    model.wf_ratio = (model.kappabar_xx / (model.conductivity_xx * model.temperature))

    ## Gamma_L
    compute_gamma_L(model, model.drude_weight_A0)
    ## Relative differences of Gamma_L
    compute_gamma_differences(model)

    ## Sigma_Q
    compute_sigmaQ(model)

    ## Shear Length \ell_\eta
    compute_entropy_A0(model)
    compute_charge_density_A0(model)
    model.shear_length = np. sqrt(model.entropy_A0 * model.conductivity_xx / ( 4*np.pi*(model.charge_density_A0**2) ))

def compute_drude_weight_A0(model):
    maskA0 = (model.lattice_amplitude == 0)
    drude_weight_A0_from_energy_pressure = model.drude_weight_from_energy_pressure[maskA0]
    # drude_weight_A0_from_temperature_entropy = model.drude_weight_from_temperature_entropy[maskA0]
    model.drude_weight_A0_from_energy_pressure = np.copy(model.drude_weight_from_energy_pressure)
    # model.drude_weight_A0_from_temperature_entropy = np.copy(model.drude_weight_from_temperature_entropy)
    for A in np.unique(model.lattice_amplitude):
        maskA = (model.lattice_amplitude == A)
        model.drude_weight_A0_from_energy_pressure[maskA] = drude_weight_A0_from_energy_pressure
        # model.drude_weight_A0_from_temperature_entropy[maskA] = drude_weight_A0_from_temperature_entropy
    model.drude_weight_A0 = model.drude_weight_A0_from_energy_pressure # since both computation are equal for A=0

def compute_gamma_L(model, drude_weight):
    model.gamma_L_from_sigma = (1 / model.conductivity_xx) * drude_weight
    # (1 / model.conductivity_xx) * (model.charge_density ** 2) / (model.energy + model.pressure)# only holds for B=0
    model.gamma_L_from_alpha = (model.s_over_rho) * (1 / model.alpha_xx) * drude_weight
    # (1 / model.alpha_xx) * (model.charge_density * model.entropy) / (model.energy + model.pressure)
    model.gamma_L_from_kappabar = model.s_over_rho**2 * model.temperature * (1 / model.kappabar_xx) * drude_weight #(model.entropy ** 2 * model.temperature / (model.charge_density ** 2)) * (1 / model.kappabar_xx) * drude_weight
    # (1 / model.kappabar_xx) * (model.entropy ** 2 * model.temperature) / (model.energy + model.pressure)

def compute_gamma_differences(model):
    model.gamma_reldiff_sigma_alpha = \
        (model.gamma_L_from_sigma - model.gamma_L_from_alpha) / model.gamma_L_from_sigma
    model.gamma_reldiff_sigma_kappabar = \
        (model.gamma_L_from_sigma - model.gamma_L_from_kappabar) / model.gamma_L_from_sigma

def compute_sigmaQ(model):
    sigma = model.conductivity_xx
    alpha = model.alpha_xx
    kappabar = model.kappabar_xx
    rho_over_s = model.rho_over_s
    T = model.temperature
    model.sigmaQ_from_sigma_alpha = \
        (sigma - (rho_over_s) * alpha) / (1 + (rho_over_s/T ))
    model.sigmaQ_from_sigma_kappabar = \
        (sigma - (rho_over_s**2 / T) * kappabar) / (1 - ( rho_over_s ** 2 / (T**2) ) )
    model.sigmaQ_from_alpha_kappabar = \
        ((rho_over_s / T) * kappabar - alpha) / ((rho_over_s / (T ** 2)) + (1 / T))

def compute_charge_density_A0(model):
    maskA0 = (model.lattice_amplitude == 0)
    charge_density_A0 = model.charge_density[maskA0]
    model.charge_density_A0 = np.copy(model.charge_density)
    for A in np.unique(model.lattice_amplitude):
        maskA = (model.lattice_amplitude == A)
        model.charge_density_A0[maskA] = charge_density_A0
def compute_entropy_A0(model):
    maskA0 = (model.lattice_amplitude == 0)
    entropy_A0 = model.entropy[maskA0]
    model.entropy_A0 = np.copy(model.entropy)
    for A in np.unique(model.lattice_amplitude):
        maskA = (model.lattice_amplitude == A)
        model.entropy_A0[maskA] = entropy_A0

def compute_s_over_rho_A0(model):
    maskA0 = (model.lattice_amplitude == 0)
    s_over_rho_A0 = model.s_over_rho[maskA0]
    model.s_over_rho = np.copy(model.s_over_rho)
    for A in np.unique(model.lattice_amplitude):
        maskA = (model.lattice_amplitude == A)
        model.s_over_rho[maskA] = s_over_rho_A0
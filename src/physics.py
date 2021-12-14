# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:02:00 2021

@author: Jonah Post
"""
import numpy as np

def calc_properties(model):
    model.kappa_xx = model.kappabar_xx - model.alpha_xx ** 2 * model.temperature / model.conductivity_xx
    # Drude Weight / omega_p squared
    model.drude_weight_from_energy_pressure = \
        (model.charge_density ** 2 / (model.energy + model.pressure))
    model.drude_weight_from_temperature_entropy = \
        (model.charge_density ** 2 / (model.entropy * model.temperature + model.charge_density))
    # Also for A=0, A-independent weight
    compute_drude_weight_A0(model)

    # Thermodynamics
    model.equation_of_state = model.energy + model.pressure - model.temperature * model.entropy - model.charge_density
    model.energy_pressure_ratio = model.energy / model.pressure
    model.one_over_mu = 1 / model.chem_pot

    # Wiedermann-Franz ratio
    model.s2_over_rho2 = (model.entropy / model.charge_density) ** 2
    model.wf_ratio = (model.kappabar_xx / (model.conductivity_xx * model.temperature))

    # Gamma_L
    compute_gamma_L(model, model.drude_weight_A0)
    # relative differences of Gamma_L
    compute_gamma_differences(model)

    # Sigma_Q
    compute_sigmaQ(model)


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
    model.gamma_L_from_alpha = (model.entropy / model.charge_density) * (1 / model.alpha_xx) * drude_weight
    # (1 / model.alpha_xx) * (model.charge_density * model.entropy) / (model.energy + model.pressure)
    model.gamma_L_from_kappabar = (model.entropy ** 2 * model.temperature / (model.charge_density ** 2)) * (
                1 / model.kappabar_xx) * drude_weight
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
    s = model.entropy
    T = model.temperature
    rho = model.charge_density
    model.sigmaQ_from_sigma_alpha = \
        (sigma - (rho / s) * alpha) / (1 + (rho / (T * s)))
    model.sigmaQ_from_sigma_kappabar = \
        (sigma - (rho ** 2 / (s ** 2 * T)) * kappabar) / (1 - (rho ** 2 / (s ** 2 * T ** 2)))
    model.sigmaQ_from_alpha_kappabar = \
        ((rho / (s * T)) * kappabar - alpha) / ((rho / (T ** 2 * s)) + (1 / T))


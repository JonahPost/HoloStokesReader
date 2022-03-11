# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:02:00 2021

@author: Jonah Post
"""
import numpy as np

def calc_properties(model):
    model.s_over_rho = model.entropy / model.charge_density
    # model.s_over_rho_A0 = compute_homogeneous_A0_value(model , "s_over_rho")
    model.rho_over_s = 1/model.s_over_rho

    model.kappa_xx = model.kappabar_xx - model.alpha_xx ** 2 *model.temperature/ model.conductivity_xx
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
    # compute_drude_weight_A0(model)

    # model.drudeweight_over_rho = model.drude_weight_from_temperature_entropy / model.charge_density

    ## Thermodynamics
    model.entropy_over_T = model.entropy / model.temperature
    model.energy_plus_pressure = model.energy + model.pressure
    model.energy_pressure_ratio = model.energy / model.pressure
    model.pressure_ratio = model.pressure_y / model.pressure_x
    model.stress_energy_trace = (-model.energy + model.pressure_x + model.pressure_y)/(model.energy)
    # model.energy_plus_pressure_A0 = compute_homogeneous_A0_value(model, "energy_plus_pressure")
    model.equation_of_state = model.energy + model.pressure - model.temperature * model.entropy - model.charge_density
    model.equation_of_state_ratio = (model.temperature * model.entropy + model.charge_density) / (model.energy + model.pressure)

    model.one_over_mu = 1 / model.chem_pot
    model.conductivity_T = model.conductivity_xx * model.temperature
    model.resistivity_xx = model.conductivity_xx / (model.conductivity_xx**2 + model.conductivity_xy**2)
    model.resistivity_over_T = model.resistivity_xx / model.temperature
    model.sigmaDC_from_amplitude = np.sqrt(3) * ( 1 + model.lattice_amplitude**2 )**2 / ( 2*np.pi*(model.lattice_amplitude**2)*np.sqrt(4 + 6*(model.lattice_amplitude**2)) * model.temperature)
    model.sigmaDC_ratio = model.conductivity_xx/model.sigmaDC_from_amplitude

    ## Wiedermann-Franz ratio
    model.s2_over_rho2 = model.s_over_rho**2
    model.wf_ratio = (model.kappabar_xx / (model.conductivity_xx * model.temperature))

    ## SIGMA_Q
    compute_sigmaQ(model)
    model.sigmaQ = model.sigmaQ_from_sigma_kappabar
    model.conductivity_drude = model.conductivity_xx - model.sigmaQ
    model.alpha_drude = model.alpha_xx + (1/model.temperature)*model.sigmaQ
    model.kappabar_drude = model.kappabar_xx - (1/model.temperature)*model.sigmaQ

    # ## SIGMA_Q ALTERNATIVE
    # model.sigmaQ_from_sigma_kappabar_new = model.conductivity_xx - (model.rho_over_s**2 / model.temperature)*model.kappabar_xx
    # model.sigmaQ_from_sigma_alpha_new = model.conductivity_xx - model.rho_over_s*model.alpha
    # model.sigmaQ = model.sigmaQ_from_sigma_kappabar_new
    # model.conductivity_drude = model.conductivity_xx - model.sigmaQ
    # model.alpha_drude = model.alpha_xx
    # model.kappabar_drude = model.kappabar_xx

    # Gamma_L
    compute_gamma_L(model, model.drude_weight_from_energy_pressure)
    ## Relative differences of Gamma_L
    compute_gamma_differences(model)



    ## Shear Length \ell_\eta
    if model.model == "EMD":
        model.entropy_A0 = compute_homogeneous_A0_value(model , "entropy")
        model.charge_density_A0 = compute_homogeneous_A0_value(model, "charge_density")
        model.shear_length      = np.sqrt(model.entropy_A0 * model.conductivity_xx / ( 4*np.pi*(model.charge_density_A0**2) ))
        # model.shear_length_alt1 = np.sqrt(model.entropy    * model.conductivity_xx / ( 4*np.pi*(model.charge_density**2   ) ))
        # model.shear_length_alt2 = np.sqrt(model.entropy_A0 * model.conductivity_xx / (4 * np.pi * model.energy_plus_pressure *model.drude_weight_A0))
        model.one_over_shear_length = 1/model.shear_length
        # model.conductivity_T_limit = model.drude_weight_A0_from_energy_pressure * 2 * np.pi**3
        # model.conductivity_T_limit = model.temperature*4*np.pi* model.charge_density_A0**2 * (np.pi/np.sqrt(2))**2 / model.entropy_A0
        model.conductivity_T_limit = model.temperature * 2 * (np.pi**3) * model.charge_density_A0 ** 2 / model.entropy_A0

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
    model.gamma_L_from_sigma = (1 / model.conductivity_drude) * drude_weight
    # (1 / model.conductivity_xx) * (model.charge_density ** 2) / (model.energy + model.pressure)# only holds for B=0
    model.gamma_L_from_alpha = (model.s_over_rho) * (1 / model.alpha_drude) * drude_weight
    # (1 / model.alpha_xx) * (model.charge_density * model.entropy) / (model.energy + model.pressure)
    model.gamma_L_from_kappabar = model.s_over_rho**2 * model.temperature * (1 / model.kappabar_drude) * drude_weight #(model.entropy ** 2 * model.temperature / (model.charge_density ** 2)) * (1 / model.kappabar_xx) * drude_weight
    # (1 / model.kappabar_xx) * (model.entropy ** 2 * model.temperature) / (model.energy + model.pressure)

def compute_gamma_differences(model):
    model.gamma_ratio_sigma_alpha = model.gamma_L_from_alpha / model.gamma_L_from_sigma
    model.gamma_ratio_sigma_kappabar = model.gamma_L_from_kappabar / model.gamma_L_from_sigma
    model.gamma_ratio_alpha_kappabar = model.gamma_L_from_kappabar / model.gamma_L_from_alpha

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

def compute_homogeneous_A0_value(model , quantity_name):
    maskA0 = (model.lattice_amplitude == 0)
    quantity_array = getattr(model, quantity_name)
    homogeneous_single_array = quantity_array[maskA0]
    homogeneous_full_array = np.copy(quantity_array)
    for A in np.unique(model.lattice_amplitude):
        maskA = (model.lattice_amplitude == A)
        homogeneous_full_array[maskA] = homogeneous_single_array
    return homogeneous_full_array
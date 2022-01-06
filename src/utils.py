# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:10:26 2021

@author: Jonah Post

Classes:
    DataSet
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class DataSet():
    def __init__(self, model, fname):
        """"
        Parameters
        ----------
        model: str
            A string specifying the model. Currently only two options: EMD or RN.
        fname: str
            A string containing the path and name to the datafile. This should be a txt file.
        """
        self.model = model
        self.filename = fname
        self.specify_model()
        self.import_data()
        # self.calc_properties()

    def specify_model(self):
        """"
        Since the different models use different keys for certain quantities, we specify them here.
        """
        if self.model == "EMD" or self.model == "emd":
            self.periodicity_key = "Gx"
            self.lattice_amplitude_key = "Ax"
            self.free_energy_key = "FreeEnergy"
            self.internal_energy_key = "InternalEnergy"
            self.chem_pot_key = "mu"
        elif self.model == "RN" or self.model == "rn":
            self.periodicity_key = "P"
            self.lattice_amplitude_key = "A0"
            self.free_energy_key = "Omega"
            self.internal_energy_key = "EInternal"
            self.chem_pot_key = "muTB"
        else:
            raise Exception("specify the model: 'EMD' or 'RN' or 'emd' or 'rn' ")

    def import_data(self):
        self.data = pd.read_csv(self.filename, sep="\t")
        self.data = self.data.sort_values(by=[self.lattice_amplitude_key, "T"])
        self.periodicity = self.data[self.periodicity_key].to_numpy()
        self.lattice_amplitude = self.data[self.lattice_amplitude_key].to_numpy()
        self.one_over_A = 1/self.lattice_amplitude
        self.temperature = self.data["T"].to_numpy()
        self.conductivity_xx = self.data["SigmaE11"].to_numpy()  # electrical conductivity
        self.conductivity_xy = self.data["SigmaE12"].to_numpy()
        self.alpha_xx = (self.data["SigmaAlpha11"].to_numpy()) / self.temperature  # thermo-electric conductivity
        self.alpha_xy = self.data["SigmaAlpha12"].to_numpy() / self.temperature
        self.alphabar_xx = self.data["SigmaAlphaBar11"].to_numpy() / self.temperature
        self.alphabar_xy = self.data["SigmaAlphaBar12"].to_numpy() / self.temperature
        self.kappabar_xx = self.data["SigmaKappa11"].to_numpy() / self.temperature  # thermal conductivity
        self.kappabar_xy = self.data["SigmaKappa12"].to_numpy() / self.temperature
        self.entropy = self.data["S"].to_numpy()
        # self.rhoH              =  self.data["rhoH"].to_numpy()
        self.energy = -self.data["Ttt"].to_numpy()  # energy stress tensor
        self.pressure = self.data["Txx"].to_numpy()
        self.pressurediffxxyy = self.data["Txx"].to_numpy() - self.data["Tyy"].to_numpy()
        try:
            self.free_energy = self.data[self.free_energy_key].to_numpy()
        except:
            pass
        try:
            self.internal_energy = self.data[self.internal_energy_key].to_numpy()
        except:
            pass
        self.charge_density = self.data["rho"].to_numpy()
        self.chem_pot = self.data[self.chem_pot_key].to_numpy()
        # self.black_hole_charge = self.data["Q"].to_numpy()

    def calc_properties(self):
        self.resistivity_xx = 1. / self.conductivity_xx
        self.resistivity_xy = 1. / self.conductivity_xy
        self.kappa_xx = self.kappabar_xx - self.alpha_xx ** 2 * self.temperature / self.conductivity_xx
        self.plasmon_frequency_squared_from_pressure = (self.charge_density ** 2 / (self.energy + self.pressure))
        self.plasmon_frequency_squared_from_temperature = (self.charge_density ** 2 / (self.entropy*self.temperature + self.charge_density))

        self.equation_of_state =self.energy + self.pressure - self.temperature*self.entropy - self.charge_density # in units of \mu=1
        self.energy_pressure_ratio = self.energy/self.pressure
        self.one_over_mu = 1/self.chem_pot
        self.s2_over_rho2 = (self.entropy/self.charge_density)**2
        self.wf_ratio = (self.kappabar_xx/(self.conductivity_xx*self.temperature))

        self.gamma_L_from_sigma =  (1/self.conductivity_xx) * (self.charge_density**2)/(self.energy + self.pressure)  # only holds for B=0
        self.gamma_L_from_alpha =  (1/self.alpha_xx) * (self.charge_density * self.entropy)/(self.energy + self.pressure)
        self.gamma_L_from_kappabar = (1 / self.kappabar_xx) * (self.entropy**2 * self.temperature) / (self.energy + self.pressure)

        self.sigmaQ_from_sigma_alpha = self.sigmaQ_from_sigma_alpha()
        self.sigmaQ_from_sigma_kappabar = self.sigmaQ_from_sigma_kappabar()
        self.sigmaQ_from_alpha_kappabar = self.sigmaQ_from_alpha_kappabar()

    def sigmaQ_from_sigma_alpha(self):
        sigma = self.conductivity_xx
        alpha = self.alpha_xx
        s = self.entropy
        T = self.temperature
        rho = self.charge_density
        return (sigma - (rho/s)*alpha ) / (1 + (rho/(T*s)))

    def sigmaQ_from_sigma_kappabar(self):
        sigma = self.conductivity_xx
        kappabar = self.kappabar_xx
        s = self.entropy
        T = self.temperature
        rho = self.charge_density
        return (sigma - (rho**2 / (s**2 * T))*kappabar) / (1 - (rho**2 / (s**2 * T**2)))

    def sigmaQ_from_alpha_kappabar(self):
        alpha = self.alpha_xx
        kappabar = self.kappabar_xx
        s = self.entropy
        T = self.temperature
        rho = self.charge_density
        return( (rho/(s*T))*kappabar - alpha) / ( (rho/(T**2 * s)) + (1/T) )

# def polynomial(x, a2, a1):
#     return a2 * (x**2) + a1 * x

def polynomial(x, a2, a1, a0):
    return a2 * (x**2) + a1 * x + a0

def pol_fit(x, y):
    """"
    Method to make a second order polynomial fit.
    """
    x_finite, y_finite = remove_nan(x,y)
    popt, pcov = curve_fit(polynomial, x_finite, y_finite)
    # popt, pcov = curve_fit(polynomial, x_finite, y_finite)
    # pol = np.poly1d(np.append(popt,0))
    pol = np.poly1d(popt)
    return np.flip(popt), pol

def remove_nan(x,y):
    finite_mask_x = np.isfinite(x)
    finite_mask_y = np.isfinite(y)
    finite_mask = (finite_mask_x*finite_mask_y)
    return x[finite_mask], y[finite_mask]

def sort(x,y):
    ind = np.argsort(x)
    return x[ind], y[ind]
    # return x,y
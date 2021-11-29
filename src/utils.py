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
        self.calc_properties()

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
        self.periodicity = self.data[self.periodicity_key].to_numpy()
        self.lattice_amplitude = self.data[self.lattice_amplitude_key].to_numpy()
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
        self.free_energy = self.data[self.free_energy_key].to_numpy()
        self.internal_energy = self.data[self.internal_energy_key].to_numpy()
        self.charge_density = self.data["rho"].to_numpy()
        self.chem_pot = self.data[self.chem_pot_key].to_numpy()
        # self.black_hole_charge = self.data["Q"].to_numpy()

    def calc_properties(self):
        self.resistivity_xx = 1. / self.conductivity_xx
        self.resistivity_xy = 1. / self.conductivity_xy
        self.kappa_xx = self.kappabar_xx - self.alpha_xx ** 2 * self.temperature / self.conductivity_xx
        self.plasma_frequency = np.sqrt(self.charge_density ** 2 / (self.energy + self.pressure))

        self.equation_of_state =self.energy + self.pressure - self.temperature*self.entropy - self.charge_density # in units of \mu=1
        self.energy_pressure_ratio = self.energy/self.pressure
        self.one_over_mu = 1/self.chem_pot
        self.wf_ratio_s2_over_rho2 = (self.entropy/self.charge_density)**2
        self.wf_ratio_kappa_over_sigmaT = (self.kappabar_xx/(self.conductivity_xx*self.temperature))

        self.gamma_L_from_sigma =  (1/self.conductivity_xx) * (self.charge_density**2)/(self.energy + self.pressure)  # only holds for B=0
        self.gamma_L_from_alpha =  (1/self.alpha_xx) * (self.charge_density * self.entropy)/(self.energy + self.pressure)
        self.gamma_L_from_kappabar = (1 / self.kappabar_xx) * (self.entropy**2 * self.temperature) / (self.energy + self.pressure)


def polynomial(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x ** 2


def pol_fit(x, y):
    """"
    Method to make a second order polynomial fit.
    """
    x_finite, y_finite = remove_nan(x,y)
    popt, pcov = curve_fit(polynomial, x_finite, y_finite)
    return popt

def remove_nan(x,y):
    finite_mask_x = np.isfinite(x)
    finite_mask_y = np.isfinite(y)
    finite_mask = (finite_mask_x*finite_mask_y)
    return x[finite_mask], y[finite_mask]

def sort(x,y):
    ind = np.argsort(x)
    return x[ind], y[ind]
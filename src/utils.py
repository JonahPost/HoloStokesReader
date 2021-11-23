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
        self.periodicity = pd.DataFrame.to_numpy(self.data[self.periodicity_key])
        self.lattice_amplitude = pd.DataFrame.to_numpy(self.data[self.lattice_amplitude_key])
        self.temperature = pd.DataFrame.to_numpy(self.data["T"])
        self.conductivity_xx = pd.DataFrame.to_numpy(self.data["SigmaE11"])  # electrical conductivity
        self.conductivity_xy = pd.DataFrame.to_numpy(self.data["SigmaE12"])
        self.alpha_xx = pd.DataFrame.to_numpy(
            self.data["SigmaAlpha11"]) / self.temperature  # thermo-electric conductivity
        self.alpha_xy = pd.DataFrame.to_numpy(self.data["SigmaAlpha12"]) / self.temperature
        self.alphabar_xx = pd.DataFrame.to_numpy(self.data["SigmaAlphaBar11"]) / self.temperature
        self.alphabar_xy = pd.DataFrame.to_numpy(self.data["SigmaAlphaBar12"]) / self.temperature
        self.kappabar_xx = pd.DataFrame.to_numpy(self.data["SigmaKappa11"]) / self.temperature  # thermal conductivity
        self.kappabar_xy = pd.DataFrame.to_numpy(self.data["SigmaKappa12"]) / self.temperature
        self.entropy = pd.DataFrame.to_numpy(self.data["S"])
        # self.rhoH              = pd.DataFrame.to_numpy(self.data["rhoH"])
        self.energy = -pd.DataFrame.to_numpy(self.data["Ttt"])  # energy stress tensor
        self.pressure = pd.DataFrame.to_numpy(self.data["Txx"])
        self.free_energy = pd.DataFrame.to_numpy(self.data[self.free_energy_key])
        self.internal_energy = pd.DataFrame.to_numpy(self.data[self.internal_energy_key])
        self.charge_density = pd.DataFrame.to_numpy(self.data["rho"])
        self.chem_pot = pd.DataFrame.to_numpy(self.data[self.chem_pot_key])
        # self.black_hole_charge = pd.DataFrame.to_numpy(self.data["Q"])

    def calc_properties(self):
        self.resistivity_xx = 1. / self.conductivity_xx
        self.resistivity_xy = 1. / self.conductivity_xy
        self.kappa_xx = self.kappabar_xx - self.alpha_xx ** 2 * self.temperature / self.conductivity_xx
        self.plasma_frequency = np.sqrt(self.charge_density ** 2 / (self.energy + self.pressure))
        self.tau_l = self.conductivity_xx / self.plasma_frequency ** 2  # only holds for B=0


def polynomial(x, b, c):
    return b * x + c * x ** 2


def pol_fit(x, y):
    """"
    Method to make a second order polynomial fit.
    """
    popt, pcov = curve_fit(polynomial, x, y)
    return popt

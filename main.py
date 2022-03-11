# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:12:45 2021

@author: Jonah Post
"""

import numpy as np
from numpy import ndarray

from src.utils import DataSet
import src.utils as utils
from src.plot_utils import *
import src.physics as physics
import matplotlib.pyplot as plt
import matplotlib
import src.IO_utils as IO_utils
from scipy.optimize import curve_fit
from matplotlib import rcParams
import palettable
import examples.New_RN_data as New_RN_data
import examples.Snellius_data as Snellius_data
import examples.Incl_integrals as Incl_integrals
import examples.planckianuniversalityplot as plankianuniversality
import examples.planckianuniversality_new as planckianuniversality_new
from examples.thermo_quantities import plot_energy_pressure, plot_conductivities, plot_entropy, plot_drude_weight, plot_universality

plt.style.use(['science','no-latex'])
rcParams['font.family'] = 'DeJavu Sans'
# rcParams['font.sans-serif'] = ['Helvetica']

SMALLER_SIZE = 4
SMALL_SIZE = 5
MEDIUM_SIZE = 7
BIGGER_SIZE = 9
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    # plankianuniversality.main()
    # New_RN_data.main()
    # Snellius_data.main()
    # Incl_integrals.main()
    planckianuniversality_new.main()
    # plot_energy_pressure(EMD)
    # plot_conductivities(EMD)
    # plot_entropy(EMD)
    # plot_drude_weight(EMD)
    # plot_universality(EMD)
    plt.show()
    plt.close()
else:
    pass


# TODO:
# clean it up.

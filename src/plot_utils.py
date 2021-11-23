# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:11:07 2021

@author: Jonah Post
"""

import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils


class QuantityQuantityPlot:
    """"
    A class to compute desired plots, for any given quantities.
    """
    def __init__(self, x_quantity_name, y_quantity_name, model_1, model_2=None, exponential=True, polynomial=False,
                 quantity_multi_line=None):
        # exponential and polynomial will only be fit to model_1
        # What happen when both model_2 and multi_line are on?
        """"
        Parameters
        ----------
        x_quantity_name: str
            The quantity that varies in value, to be used on the horizontal axis.
            Can be one of: "temperature", "periodicity", "lattice_amplitude".
        y_quantity_name: str
            Quantity to be plotted on the vertical axis
        model_1: utils.DataSet
            The main dataset to be plotted.
        model_2: utils.DataSet
            If given, the second dataset will be plotted along with the first.
            Also a legend will be included and the title will change appropriately.
        exponential: bool
            If True, a extra subplot will be added, plotting the the exponent of an exponential "fit" to model_1.
            This is actually not a fit, but the coefficient dlog(y)/dlog(x).
        polynomial: bool
            If Ture, a second order polynomial fit to the data of model_1 in the first subplot.
            Also legend will be included.
        quantity_multi_line: str
            The name of the quantity of which multiple lines will be drawn for different values of this quantity.
        """
        self.dict = {"temperature": "T", "periodicity": "G", "lattice_amplitude": "A"}
        self.x_quantity_name = x_quantity_name
        self.y_quantity_name = y_quantity_name
        self.model_1 = model_1
        self.model_2 = model_2
        self.exponential = exponential
        self.polynomial = polynomial
        self.quantity_multi_line = quantity_multi_line
        if self.quantity_multi_line is not None:
            self.multiline_data = getattr(self.model_1, self.quantity_multi_line)

        # read model data
        self.xdata1 = getattr(model_1, x_quantity_name)
        self.ydata1 = getattr(model_1, y_quantity_name)
        if self.model_2 is not None:
            self.xdata2 = getattr(self.model_2, self.x_quantity_name)
            self.ydata2 = getattr(self.model_2, self.y_quantity_name)

        # Initialize figure
        n_figs = 1 + int(exponential)  # if necessary add more terms here
        if n_figs == 1:
            self.fig, self.ax1 = plt.subplots(n_figs, 1, sharex='all')
        else:
            self.fig, self.axs = plt.subplots(n_figs, 1, sharex='all')
            self.ax1 = self.axs[0]

        self.compute_title_prefix()
        self.compute_title_appendix()
        self.title1 = self.title_prefix + f"{y_quantity_name} vs {x_quantity_name}" + ", " + self.title_appendix
        self.ax1.set_title(self.title1)
        self.ax1.set_ylabel(y_quantity_name)
        self.plot_lines()

    def plot_lines(self):
        """"
        The method actually plots the data
        """
        if self.quantity_multi_line is not None:
            for quantity_value in np.unique(self.multiline_data):
                mask = (self.multiline_data == quantity_value)
                line_label = self.dict[self.quantity_multi_line] + "=" + str(quantity_value)
                self.ax1.plot(self.xdata1[mask], self.ydata1[mask], "-x", label=self.label_prefix + line_label)
        else:
            self.ax1.plot(self.xdata1, self.ydata1, "-x", label=self.label_prefix+"data")

        # If you want to compare with model_2 data
        if self.model_2 is not None:
            self.ax1.plot(self.xdata2, self.ydata2, "-x", label=self.model_2.model)

        # Any additional fits etc.
        if self.exponential == True:
            self.ax_exp = self.axs[1]
            self.ax_exp.set_ylabel("exponent")
            self.ax_exp.set_xlabel(self.x_quantity_name)
            self.ax_exp.set_title(
                f"{self.model_1.model} {self.y_quantity_name} exponent" + ", " + self.title_appendix)
            self.ax_exp.plot(self.xdata1, np.gradient(np.log(self.ydata1), self.xdata1) * self.xdata1, "-x",
                             label=self.label_prefix + "exponent")
        else:
            self.ax1.set_xlabel(self.x_quantity_name)
        if self.polynomial == True:
            popt = utils.pol_fit(self.xdata1, self.ydata1)
            xrange = np.linspace(self.xdata1[0], self.xdata1[-1])
            self.ax1.plot(xrange, utils.polynomial(xrange, *popt), label=r"{:.2f}$x$+{:.2f}$x^2$".format(*popt))
        self.ax1.legend()

    def compute_title_appendix(self):
        other_x_quantities = ["temperature", "periodicity", "lattice_amplitude"]
        other_x_quantities.remove(self.x_quantity_name)
        if self.quantity_multi_line is not None:
            other_x_quantities.remove(self.quantity_multi_line)
        self.title_appendix = ""
        for quantity in other_x_quantities:
            self.title_appendix += f" {self.dict[quantity]}={getattr(self.model_1, quantity)[0]:.2f}"

    def compute_title_prefix(self):
        if self.model_2 is not None:
            self.title_prefix = ""
            self.label_prefix = self.model_1.model + " "
        else:
            self.title_prefix = self.model_1.model + " "
            self.label_prefix = ""

    def savefig(self, *args, **kwargs):
        """"
        A method to save the figure. A redirect to the plt.savefig, but can now be applied to a utils.DataSet object.
        """
        self.fig.savefig(*args, **kwargs)

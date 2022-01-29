# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:11:07 2021

@author: Jonah Post
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
import src.utils as utils
# plt.style.reload_library()
plt.style.use(['science','ieee','no-latex'])

class QuantityQuantityPlot:
    """"
    A class to compute desired plots, for any given quantities.
    """

    def __init__(self, x_quantity_name, y_quantity_name, model_1, model_2=None, exponential=False, polynomial=False,
                 quantity_multi_line=None, mask1=None, mask2=None, logy=False, logx=False, fname_appendix="", scienceplot=True, linelabels=False, cbar=True, ax= None, figure=None):
        # exponential and polynomial will only be fit to model_1
        # What happen when both model_2 and multi_line are on?
        """"
        Parameters
        ----------
        x_quantity_name: str
            The quantity that varies in value, to be used on the horizontal axis. Can be one of: "temperature",
            "periodicity", "lattice_amplitude".
        y_quantity_name: str
            Quantity to be plotted on the vertical axis
        model_1: utils.DataSet
            The main dataset to be plotted.
        model_2: utils.DataSet
            If given, the second dataset will be plotted along with the first. Also a legend will be included and the
            title will change appropriately.
        exponential: bool
            If True, a extra subplot will be added, plotting the the exponent of an exponential "fit" to model_1. This
            is actually not a fit, but the coefficient dlog(y)/dlog(x).
        polynomial: bool
            If Ture, a second order polynomial fit to the data of model_1 in the first subplot. Also legend will be
            included.
        quantity_multi_line: str
            The name of the quantity of which multiple lines will be drawn for different values of this quantity.
        mask: ndarray
            An arraay used to mask parts of the data. For example if you don't want to plot A=0, then you should add a
            mask which excludes this data, e.g. a (A!=0) mask.
        loglog: bool
            Boolean value for whether or not to make the plots on a loglog scale.
        """
        self.long_dict = {
            "temperature": r"$T/\mu$",
            "periodicity": "Periodicity",
            "lattice_amplitude": "Amplitude",
            "one_over_A": r"$1/A$",
            "entropy": "Entropy"
            }
        self.short_dict = {
            "temperature": r"$T$",
            "periodicity": r"$G$",
            "lattice_amplitude": r"$A$",
            "energy": r"$\mathcal{E}$",
            "pressure": r"$\mathcal{P}$",
            "energy_plus_pressure": r"$\mathcal{E}+\mathcal{P}$",
            "energy_pressure_ratio": r"$\mathcal{E}/\mathcal{P}$",
            "one_over_A": r"$\frac{1}{A}$",
            "entropy": r"$S$",
            "entropy_over_T": r"$S/T$",
            "charge_density": r"$\rho$",
            "gamma_L_from_sigma": r"$\Gamma_{L,\sigma}$",
            "gamma_L_from_alpha": r"$\Gamma_{L,\alpha}$",
            "gamma_L_from_kappabar": r"$\Gamma_{L,\bar{\kappa}}$",
            "drude_weight_from_energy_pressure": r"$\omega_p^2 (\mathcal{E}+\mathcal{P})$",
            "drude_weight_from_temperature_entropy": r"$\omega_p^2 (S, T)$",
            "drude_weight_A0": r"$\omega_p^2 (A=0)$",
            "conductivity_xx": r"$\sigma_{xx}$",
            "alpha_xx": r"$\alpha_{xx}$",
            "kappa_xx": r"$\kappa_{xx}$",
            "kappabar_xx": r"$\bar{\kappa}_{xx}$",
            "sigmaQ_from_sigma_alpha": r"$\sigma_Q$",
            "sigmaQ_from_sigma_kappabar": r"$\sigma_Q$",
            "sigmaQ_from_alpha_kappabar": r"$\sigma_Q$",
            "shear_length": r"$\ell_{\eta}$",
            "shear_length_alt1": r"$\ell_{\eta}$",
            "shear_length_alt2": r"$\ell_{\eta}$",
            "one_over_shear_length": r"$q_{\eta}=\frac{1}{\ell_{\eta}}$",
            "resistivity_over_T": r"$\rho/T$",
            "resistivity_xx": r"$\rho$",
            "conductivity_T": r"$\sigma T$",
        }
        self.scienceplot = scienceplot
        self.ax = ax
        self.fig = figure
        self.linelabels = linelabels
        self.cbar = cbar
        self.x_quantity_name = x_quantity_name
        self.y_quantity_name = y_quantity_name
        self.model_1 = model_1
        self.model_2 = model_2
        self.exponential = exponential
        self.polynomial = polynomial
        self.quantity_multi_line = quantity_multi_line
        self.mask1 = mask1
        self.mask2 = mask2
        self.fname_appendix = fname_appendix
        # read model data
        self.xdata1 = getattr(self.model_1, self.x_quantity_name)[self.mask1]
        self.ydata1 = getattr(self.model_1, self.y_quantity_name)[self.mask1]
        if self.quantity_multi_line is not None:
            self.multiline_data1 = getattr(self.model_1, self.quantity_multi_line)[self.mask1]
        if self.model_2 is not None:
            self.xdata2 = getattr(self.model_2, self.x_quantity_name)[self.mask2]
            self.ydata2 = getattr(self.model_2, self.y_quantity_name)[self.mask2]
            if self.quantity_multi_line is not None:
                self.multiline_data2 = getattr(self.model_2, self.quantity_multi_line)[self.mask2]

        # Initialize figure
        if self.ax is not None:
            self.ax1 = self.ax
            if self.fig is None:
                raise Exception("If you give ax, you must also also give fig")
        else:
            figure_size = (4, 2)
            self.fig, self.ax1 = plt.subplots(1, 1, figsize=figure_size)

        self.compute_title_prefix()
        self.compute_title_appendix()
        self.title1 = (self.title_prefix + f"{y_quantity_name} vs {x_quantity_name}" + self.title_appendix).replace("_", " ")
        if not scienceplot:
            self.ax1.set_title(self.title1)
        # if scienceplot:
        #     self.ax1.text(.95,.95, self.title_appendix,
        #             horizontalalignment='right',
        #             verticalalignment='top',
        #             transform=self.ax1.transAxes)
        if y_quantity_name in self.short_dict.keys():
            self.ax1.set_ylabel(self.short_dict[y_quantity_name])
        else:
            self.ax1.set_ylabel(y_quantity_name)
        self.ax1.set_xlabel(self.long_dict[self.x_quantity_name])
        if logx:
            self.ax1.set_xscale("log")
        if logy:
            self.ax1.set_yscale("log")
        if self.exponential:
            self.fig_exp, self.ax_exp = plt.subplots(1, 1, figsize=figure_size)
            self.ax_exp.grid()
            self.ax_exp.set_ylabel("exponent")
            self.ax_exp.set_xlabel(self.long_dict[self.x_quantity_name])
            self.ax_exp.set_title(
                f"{self.model_1.model}: {self.y_quantity_name} exponent" + self.title_appendix)
        self.plot_lines()
        if self.exponential:
            self.fig_exp.tight_layout()

    def plot_lines(self):
        """"
        The method actually plots the data
        """
        if self.quantity_multi_line is not None:
            self.make_cbar()
            for i, quantity_value in enumerate(np.unique(self.multiline_data1)):
                mask = (self.multiline_data1 == quantity_value)
                x, y = utils.sort(self.xdata1[mask], self.ydata1[mask])
                x, y = utils.remove_nan(x, y)
                if self.linelabels:
                    line_label = self.short_dict[self.quantity_multi_line] + "={:.2f}".format(quantity_value)
                else:
                    line_label=None
                line_color = self.cmap1.to_rgba(quantity_value)
                if self.scienceplot:
                    # self.ax1.plot(x, y, "-", label=self.label_prefix1 + line_label, c=line_color)
                    self.ax1.plot(x, y, "-", markersize = 3, c=line_color, label=line_label)
                else:
                    self.ax1.plot(x, y, "-x", label=self.label_prefix1 + line_label, c=line_color)
                if self.exponential:
                    self.ax_exp.plot(x, np.gradient(np.log(y), x) * x, "-x",
                                     label=self.label_prefix1 + line_label, c=line_color)
                if self.polynomial:
                    popt, pol = utils.pol_fit(x, y)
                    xrange = np.linspace(0, x[-1])
                    # self.ax1.plot(xrange, pol(xrange), "--", label=r"{:.2g}$x$+{:.2g}$x^2$".format(*popt), c="k")
                    self.ax1.plot(xrange, pol(xrange), "--", label=r"{:.2g}+{:.2g}$x$".format(*popt), c="k")
            if self.model_2 is not None:
                for i, quantity_value in enumerate(np.unique(self.multiline_data2)):
                    mask = (self.multiline_data2 == quantity_value)
                    x2, y2 = utils.sort(self.xdata2[mask], self.ydata2[mask])
                    line_label = self.short_dict[self.quantity_multi_line] + "=" + str(quantity_value)
                    line_color = self.cmap2.to_rgba(quantity_value)
                    self.ax1.plot(x2, y2, "-x", label=self.label_prefix2 + line_label, c=line_color)
                    if self.exponential:
                        self.ax_exp.plot(x2, np.gradient(np.log(y2), x2) * x2, "-x",
                                         label=self.label_prefix2 + line_label, c=line_color)
                if self.cbar:
                    self.cbar2 = self.fig.colorbar(self.cmap2, ax=self.ax1)
                    self.cbar2.set_label(self.label_prefix2 + self.long_dict[self.quantity_multi_line])
                    if self.exponential:
                        self.cbarexp2 = self.fig.colorbar(self.cmap2, ax=self.ax_exp)
                        self.cbarexp2.set_label(self.label_prefix2 + self.dict[self.quantity_multi_line])
            if self.cbar:
                self.cbar1 = self.fig.colorbar(self.cmap1, ax=self.ax1)
                self.cbar1.set_label(self.label_prefix1 + self.long_dict[self.quantity_multi_line])
                if self.exponential:
                    self.cbarexp1 = self.fig.colorbar(self.cmap1, ax=self.ax_exp)
                    self.cbarexp1.set_label(self.label_prefix1 + self.long_dict[self.quantity_multi_line])
                if self.polynomial:
                    self.cbar1.remove()
                    if self.model_2 is not None:
                        self.cbar2.remove()
                    self.ax1.legend(loc=(1, 0))

        else:
            x, y = utils.sort(self.xdata1, self.ydata1)
            self.ax1.plot(x, y, "-x", label=self.label_prefix1 + "data")
            x2, y2 = utils.sort(self.xdata2, self.ydata2)
            self.ax1.plot(x2, y2, "-x", label=self.label_prefix2 + "data")
            if self.exponential:
                self.ax_exp.plot(self.xdata1, np.gradient(np.log(self.ydata1), self.xdata1) * self.xdata1, "-x",
                                 label=self.label_prefix1 + "exponent")
            if self.polynomial:
                popt, pol = utils.pol_fit(self.xdata1, self.ydata1)
                xrange = np.linspace(self.xdata1[0], self.xdata1[-1])
                self.ax1.plot(xrange, pol(xrange), label=r"{:.2f}+{:.2f}$x$+{:.2f}$x^2$".format(*popt))
            self.ax1.legend()
        if not np.any(self.ydata1 < 0):
            self.ax1.set_ylim(ymin=0)
        self.ax1.set_xlim(xmin=0)
        # self.ax1.grid()

    def compute_title_appendix(self):
        other_x_quantities = ["temperature", "periodicity", "lattice_amplitude"]
        if self.x_quantity_name == "one_over_A":
            other_x_quantities = ["temperature", "periodicity", "one_over_A"]
        other_x_quantities.remove(self.x_quantity_name)
        if self.quantity_multi_line is not None:
            other_x_quantities.remove(self.quantity_multi_line)
        self.title_appendix = ""
        for quantity in other_x_quantities:
            self.title_appendix += f" {self.short_dict[quantity]}={getattr(self.model_1, quantity)[0]:.2f}"

    def compute_title_prefix(self):
        if self.model_2 is not None:
            self.title_prefix = ""
            self.label_prefix1 = self.model_1.model + "- "
            self.label_prefix2 = self.model_2.model + "- "
        else:
            self.title_prefix = self.model_1.model + "- "
            self.label_prefix1 = ""

    def savefig(self, foldername, *args, **kwargs):
        """"
        A method to save the figure. A redirect to the plt.savefig, but can now be applied to a utils.DataSet object.
        """
        path = foldername + "/"
        fname = path + self.title1.replace(" ", "_")
        # if self.exponential:
        #     fname += "_exp"
        if self.polynomial:
            fname += "_polyfit"
        fname_append = "_" + self.fname_appendix + ".png"
        self.fig.savefig(fname + fname_append, *args, **kwargs)
        print("fig saved: " + fname + fname_append)
        if self.exponential:
            self.fig_exp.savefig(fname + "_exp" + fname_append, *args, **kwargs)
            print("fig saved: " + fname + "_exp" + fname_append)

    def make_cbar(self):
        quantities = np.unique(self.multiline_data1)
        if quantities.min() < 0:
            norm1 = mpl.colors.Normalize(vmin=quantities.min(), vmax=quantities.max())
        else:
            norm1 = mpl.colors.Normalize(vmin=0, vmax=quantities.max())
        # self.cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.winter.reversed())
        self.cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.inferno.reversed())
        # if self.model_1.model == "RN" or self.model_1.model == "rn":
        #     self.cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.Wistia)
        self.cmap1.set_array([])
        if self.model_2 is not None:
            quantities = np.unique(self.multiline_data2)
            norm2 = mpl.colors.Normalize(vmin=quantities.min(), vmax=quantities.max())
            self.cmap2 = mpl.cm.ScalarMappable(norm=norm2, cmap=mpl.cm.Wistia)
            self.cmap2.set_array([])
        if self.polynomial is True:
            self.cmap_poly = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.spring)
            self.cmap_poly.set_array([])

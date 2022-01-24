import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','no-latex'])
import matplotlib as mpl
from scipy import optimize
from datetime import datetime
import os
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

def plot_RN(ax_conductivity, ax_shearlength):
    ax = [ax_conductivity, ax_shearlength]

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    path = "data/" # the folder where the data files are located
    AT_fname = "ATSeriesHighA.txt"
    BT_A0_fname = "BTSeriesA0.txt"

    def divide_T(df):
        # Divide out the extra T-factor

        T = df["T"]

        df["SigmaAlpha11"] = df["SigmaAlpha11"] / T
        df["SigmaAlpha12"] = df["SigmaAlpha12"] / T
        df["SigmaAlpha21"] = df["SigmaAlpha21"] / T
        df["SigmaAlpha22"] = df["SigmaAlpha22"] / T

        df["SigmaAlphaBar11"] = df["SigmaAlphaBar11"] / T
        df["SigmaAlphaBar12"] = df["SigmaAlphaBar12"] / T
        df["SigmaAlphaBar21"] = df["SigmaAlphaBar21"] / T
        df["SigmaAlphaBar22"] = df["SigmaAlphaBar22"] / T

        df["SigmaKappa11"] = df["SigmaKappa11"] / T
        df["SigmaKappa12"] = df["SigmaKappa12"] / T
        df["SigmaKappa21"] = df["SigmaKappa21"] / T
        df["SigmaKappa22"] = df["SigmaKappa22"] / T

        return df

    def change_kappa(df):
        #Calculate Kappa from KappaBar

        T = df["T"]

        df = df.rename(columns={"SigmaKappa11": "SigmaKappaBar11", "SigmaKappa12": "SigmaKappaBar12",\
                               "SigmaKappa21": "SigmaKappaBar21", "SigmaKappa22": "SigmaKappaBar22"})

        df["SigmaKappa11"] = np.zeros(len(df["SigmaKappaBar11"]))
        df["SigmaKappa12"] = np.zeros(len(df["SigmaKappaBar12"]))
        df["SigmaKappa21"] = np.zeros(len(df["SigmaKappaBar21"]))
        df["SigmaKappa22"] = np.zeros(len(df["SigmaKappaBar22"]))

        for i in np.arange(1,len(T)+1):
            KappaBarMat = np.array([[df["SigmaKappaBar11"][i], df["SigmaKappaBar12"][i]],\
                                   [df["SigmaKappaBar21"][i], df["SigmaKappaBar22"][i]]])

            AlphaMat = np.array([[df["SigmaAlpha11"][i], df["SigmaAlpha12"][i]],\
                                   [df["SigmaAlpha21"][i], df["SigmaAlpha22"][i]]])

            SigmaMat = np.array([[df["SigmaE11"][i], df["SigmaE12"][i]],\
                                   [df["SigmaE21"][i], df["SigmaE22"][i]]])

            invSigmaMat = np.linalg.inv(SigmaMat)
            DotAlpSigm = np.dot(AlphaMat, invSigmaMat)
            KappaMat = KappaBarMat - np.dot(DotAlpSigm, AlphaMat)*T[i]

            df["SigmaKappa11"][i] = KappaMat[0,0]
            df["SigmaKappa12"][i] = KappaMat[0,1]
            df["SigmaKappa21"][i] = KappaMat[1,0]
            df["SigmaKappa22"][i] = KappaMat[1,1]

        return df

    #-----------------------------

    df = pd.read_csv(path+AT_fname, sep = '\t')
    df = divide_T(df)
    #df = change_kappa(df)
    #df = df.sort_values("T")

    dfT005 = df.loc[(df["T"]>0.0049) & (df["T"]<0.0051)]
    dfT005 = dfT005.sort_values("A0")

    dfT01 = df.loc[(df["T"]>0.009) & (df["T"]<0.011)]
    dfT01 = dfT01.sort_values("A0")

    dfT02 = df.loc[(df["T"]>0.019) & (df["T"]<0.021)]
    dfT02 = dfT02.sort_values("A0")

    dfT03 = df.loc[(df["T"]>0.029) & (df["T"]<0.031)]
    dfT03 = dfT03.sort_values("A0")

    dfT04 = df.loc[(df["T"]>0.039) & (df["T"]<0.041)]
    dfT04 = dfT04.sort_values("A0")

    dfT05 = df.loc[(df["T"]>0.049) & (df["T"]<0.051)]
    dfT05 = dfT05.sort_values("A0")

    dfT06 = df.loc[(df["T"]>0.059) & (df["T"]<0.061)]
    dfT06 = dfT06.sort_values("A0")

    dfT07 = df.loc[(df["T"]>0.069) & (df["T"]<0.071)]
    dfT07 = dfT07.sort_values("A0")

    dfT08 = df.loc[(df["T"]>0.079) & (df["T"]<0.081)]
    dfT08 = dfT08.sort_values("A0")

    dfT09 = df.loc[(df["T"]>0.089) & (df["T"]<0.091)]
    dfT09 = dfT09.sort_values("A0")

    dfT1 = df.loc[(df["T"]>0.099) & (df["T"]<0.101)]
    dfT1 = dfT1.sort_values("A0")

    # -----------------------------

    df = pd.read_csv(path+BT_A0_fname, sep='\t')
    df = divide_T(df)
    # df = change_kappa(df)
    # df = df.sort_values("T")
    df["B/mu^2"] = df["B"] / df["muTB"] ** 2

    dfB0 = df.loc[(df["B/mu^2"] > -0.000001) & (df["B/mu^2"] < 0.00001)]
    dfB0 = dfB0.sort_values("T")
    dfB0 = dfB0.reset_index(drop=True)

    # -----------------------------

    norm1 = mpl.colors.Normalize(vmin=0, vmax=0.10)
    cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.inferno.reversed())
    cmap1.set_array([])

    # -----------------------------

    df_list = [dfT01, dfT03, dfT06, dfT1]
    T_list = ["0.01", "0.03", "0.06", "0.10"]

    count = 0
    for df in df_list:
        # df = divide_T(df)
        # df = change_kappa(df)
        df = df.sort_values("A0")

        df = df[1:5]

        T = df["T"]
        G = df["P"]
        A = df["A0"]
        print(A)

        omega_P_squared = df["rho"] ** 2 / (-df["Ttt"] + df["Txx"])

        # B=0 so:
        sigma_xx = df["SigmaE11"]
        tauL = sigma_xx / omega_P_squared

        B = df["B"] / df["muTB"] ** 2
        Hall_angle = df["SigmaE12"] / df["SigmaE11"]
        omega_c = df["rho"] * B / (-df["Ttt"] + df["Txx"])
        tauT = Hall_angle / omega_c

        muTB = df["muTB"]
        Mu = df["Mu"]

        # -----------------------------

        df = dfB0

        T = df["T"][0]
        S = df["S"][0]
        epsilon = -df["Ttt"][0]
        P = df["Txx"][0]

        # -----------------------------

        tauL_sat = 2 * np.pi ** 3 * (1 / 1) * (epsilon + P) / S
        rho = df["rho"]
        plasmon_RN = rho ** 2 / (epsilon + P)
        sigma_xx_sat = plasmon_RN * tauL_sat

        y = 1 / (4 * np.pi) * S / (epsilon + P) * tauL
        y = np.sqrt(y)

        A = 1 / A

        # For inset plot
        y_sub = y
        A_sub = A


        def test_func(x, a, b):
            return a * x + b


        y_test = y[:2]
        A_test = A[:2]

        params, params_covariance = optimize.curve_fit(test_func, A_test, y_test)
        x = np.arange(0, 2, 0.001)


        def fit_func(x):
            return params[0] * x + params[1]


        x_sub = x
        fit_sub = fit_func(x)

        # plt.plot(x, fit_func(x), '--', color='k', label=r"Fit: " + str(round(params[0], 2)) + "$ \cdot$ (1/A) + " + str(round(params[1], 2)))
        ax[1].plot(x_sub, fit_sub, linestyle=(0, (1, 1)), c="k", linewidth=.5)
        line_color = cmap1.to_rgba(float(T_list[count]))
        ax[1].plot(A_sub, y_sub, '-', color=line_color, label=(r"$T={}$".format(float(T_list[count]))))

        count = count + 1

    x = np.arange(0, 2, 0.001)
    x_sub2 = x

    # Since Mu = 1:
    ones = np.ones(len(x))
    y = np.pi / (np.sqrt(2) * ones)

    # For inset plot
    line_sub = y

    # -----------------------------

    df = pd.read_csv(path+AT_fname, sep = '\t')
    df = divide_T(df)
    #df = change_kappa(df)
    #df = df.sort_values("T")

    dfA01 = df.loc[(df["A0"]>0.09) & (df["A0"]<0.11)]
    dfA01 = dfA01.sort_values("T")

    dfA05 = df.loc[(df["A0"]>0.49) & (df["A0"]<0.51)]
    dfA05 = dfA05.sort_values("T")

    dfA1 = df.loc[(df["A0"]>0.99) & (df["A0"]<1.01)]
    dfA1 = dfA1.sort_values("T")

    dfA15 = df.loc[(df["A0"]>1.49) & (df["A0"]<1.51)]
    dfA15 = dfA15.sort_values("T")

    dfA2 = df.loc[(df["A0"]>1.99) & (df["A0"]<2.01)]
    dfA2 = dfA2.sort_values("T")

    dfA25 = df.loc[(df["A0"]>2.49) & (df["A0"]<2.51)]
    dfA25 = dfA25.sort_values("T")

    dfA3 = df.loc[(df["A0"]>2.99) & (df["A0"]<3.01)]
    dfA3 = dfA3.sort_values("T")

    # -----------------------------

    norm0 = mpl.colors.Normalize(vmin=0, vmax=2)
    cmap0 = mpl.cm.ScalarMappable(norm=norm0, cmap=mpl.cm.inferno.reversed())
    cmap0.set_array([])

    # -----------------------------

    df_list = [dfA05, dfA1, dfA15, dfA2]
    A_list = ["0.5", "1.0", "1.5", "2.0"]

    count = 0

    for df in df_list:
        df = df[4:]

        T = df["T"]
        G = df["P"]
        A = df["A0"]

        # B=0 so:
        sigma_xx = df["SigmaE11"]
        y = sigma_xx

        # line_color = cmap.to_rgba(float(A_list[count]))
        line_color = cmap0.to_rgba(float(A_list[count]))
        ax[0].plot(T[1:], y[1:], '-', c=line_color)

        count = count + 1

    # -----------------------------


    ax[0].set_ylim([0, 75])
    ax[0].set_xlabel(r"$T/\mu$")
    ax[0].set_ylabel(r"$\sigma_{xx}$")
    ax[0].annotate("B", xy=(0.03, 0.91), xycoords="axes fraction", fontsize=7, fontweight='bold')

    # -----------------------------

    ax[1].plot(x_sub2, line_sub, linestyle="--", color="r", label=r"$\pi / (\mu\sqrt{2})$")
    ax[1].set_ylim([0, 8])
    ax[1].set_xlabel(r"$1/A$")
    ax[1].set_ylabel(r"$\ell_{\eta}$")
    ax[1].annotate("D", xy=(0.03, 0.91), xycoords="axes fraction", fontsize=7, fontweight='bold')

    # -----------------------------

    # All for G/mu = 0.1 and B/mu^2 = 0

if __name__ == '__main__':
    figure_size = (8, 4)
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=figure_size)
    plot_RN(ax2, ax4)
    ax2.set_xlim(xmin=0, xmax=0.10)
    ax4.set_xlim(xmin=0, xmax=2)
    ax4.set_ylim(ymin=0, ymax=10)
    ax4.legend()
    fig.tight_layout()

    if not os.path.isdir('plots'):
        os.mkdir("plots")
    folder_name = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    os.mkdir(folder_name)
    print("new folder made")
    fig.savefig(folder_name + "/PlanckianUniversality.png")
    print("plots are saved")

    plt.show()
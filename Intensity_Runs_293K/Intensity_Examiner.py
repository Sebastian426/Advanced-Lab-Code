import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
plt.rcParams["font.family"] = "Times New Roman"


###########################

#Setup Functions

###########################

def wavelength_extraction(x, start_index):
    A = 1.0095
    alpha_A = 0.0001

    B = 9091.0570
    alpha_B = 0.0289

    wavelengths = []
    uncertainties = []

    for i in range(len(x)):
        wavelength = (start_index - x[i] / 10 - B) / A
        uncertainty = np.abs((x[i] - (B + alpha_B)) / (A + alpha_A) - wavelength)

        wavelengths.append(wavelength)
        uncertainties.append(uncertainty)

    return wavelengths

def lorentzian_function(x, A, x0, B, C):
    B = np.abs(B)
    return A / (1 + ((x - x0)/B)**2) + C


def chi2_function(model, model_params, x_data, y_data, y_error):
    return np.sum(((y_data - model(x_data, *model_params)) / y_error)**2)

def rchi2_function(model, model_params, x_data, y_data, y_error, DoF):
    return np.sum(((y_data - model(x_data, *model_params)) / y_error)**2)/DoF


###########################

#Pipeline

###########################

def intensity_examiner(filename, sheet_name, column_name, start_index):

    results = {}

    for s in range(len(sheet_name)):

        sheet = sheet_name[s]
        cols = column_name[s]
        start = start_index[s]

        print(f"\nProcessing sheet: {sheet}")

        df = pd.read_excel(filename, sheet_name=sheet)

        # ---------- WAVELENGTH EXTRACTION ----------
        indexes = np.arange(0, len(df[cols[0]]), 1)
        wavelengths = wavelength_extraction(indexes, start)

        wavelengths = np.array(wavelengths, dtype=float)

        # ---------- PLOT ALL SPECTRA TOGETHER ----------
        plt.figure(1)
        for col in cols:
            plt.plot(wavelengths, df[col], label=col)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.minorticks_on()
        plt.tick_params(axis='both', which='major', size=6, direction='in')
        plt.tick_params(axis='both', which='minor', size=3, direction='in')
        plt.legend()
        plt.show()

        # ---------- FITTING ----------
        x0_list = []

        for col in cols:

            ydata = df[col].values.astype(float)

            y_fitdata = []
            wavelengths_chosen = []

            cutoff = 0.1

            for i in range(len(ydata)):
                if ydata[i] > cutoff * ydata.max():
                    y_fitdata.append(ydata[i])
                    wavelengths_chosen.append(wavelengths[i])

            y_fitdata = np.array(y_fitdata, dtype=float)
            wavelengths_chosen = np.array(wavelengths_chosen, dtype=float)

            if len(y_fitdata) < 5:
                print(f"Skipping {col}: not enough data after cutoff.")
                continue

            half_max = ydata.max() / 2
            indices = np.where(ydata > half_max)[0]

            if len(indices) > 1:
                B_guess = (wavelengths[indices[-1]] - wavelengths[indices[0]]) / 2
            else:
                B_guess = (wavelengths.max() - wavelengths.min()) / 10

            p0_lorentzian = [
                ydata.max(),           # A
                693,                   # x0 (your original guess)
                B_guess,    # B
                y_fitdata.min()        # C
            ]

            sigma = np.full_like(y_fitdata, y_fitdata.max()/100, dtype=float)

            popt, cov = curve_fit(
                lorentzian_function,
                xdata=wavelengths_chosen,
                ydata=y_fitdata,
                sigma=sigma,
                absolute_sigma=True,
                p0=p0_lorentzian,
                maxfev=10000
            )

            dof = len(y_fitdata) - len(popt)

            rchi2 = rchi2_function(
                lorentzian_function,
                popt,
                wavelengths_chosen,
                y_fitdata,
                sigma,
                dof
            )

            x0_list.append(popt[1])

            # ---------- INDIVIDUAL FIT PLOT ----------
            plt.figure(1)
            plt.scatter(wavelengths_chosen, y_fitdata, color="black", s=3)
            plt.plot(
                wavelengths_chosen,
                lorentzian_function(wavelengths_chosen, *popt),
                color="grey"
            )
            plt.errorbar(
                wavelengths_chosen,
                y_fitdata,
                sigma,
                linestyle="none",
                capsize=2,
                color="black"
            )

            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")

            plt.minorticks_on()
            plt.tick_params(axis='both', which='major', size=6, direction='in')
            plt.tick_params(axis='both', which='minor', size=3, direction='in')
            plt.show()

            print(" ----- Results -----")
            print(f"x0: {popt[1]}")
            print(f"Reduced Chi-Squared: {rchi2}")
            print("----- ----- -----")

        # ---------- SHEET SUMMARY ----------
        mean_x0 = np.mean(x0_list)
        err_x0 = np.std(x0_list) / np.sqrt(len(x0_list))

        print(f"\nMean x0: {mean_x0:.4f} +- {err_x0:.4f}")

        results[sheet] = {
            "mean_x0": mean_x0,
            "err_x0": err_x0,
            "x0_values": x0_list
        }

    return results

def double_lorentzian(x, A1, x01, B1, A2, x02, B2, C):
    return (
        A1 / (1 + ((x - x01)/B1)**2)
        + A2 / (1 + ((x - x02)/B2)**2)
        + C
    )


def double_lorentzian_intensity_extractor(filename, sheet_name, column_name, start_index):

    results = {}

    for s in range(len(sheet_name)):

        sheet = sheet_name[s]
        cols = column_name[s]
        start = start_index[s]

        print(f"\nProcessing sheet: {sheet}")

        df = pd.read_excel(filename, sheet_name=sheet)

        # ---------- WAVELENGTH EXTRACTION ----------
        indexes = np.arange(0, len(df[cols[0]]), 1)
        wavelengths = wavelength_extraction(indexes, start)
        wavelengths = np.array(wavelengths, dtype=float)

        # ---------- PLOT ALL SPECTRA ----------
        plt.figure(1)
        for col in cols:
            plt.plot(wavelengths, df[col], label=col)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.minorticks_on()
        plt.tick_params(axis='both', which='major', size=6, direction='in')
        plt.tick_params(axis='both', which='minor', size=3, direction='in')
        plt.legend()
        plt.show()

        x01_list = []
        x02_list = []

        # ---------- FIT EACH SPECTRUM ----------
        for col in cols:

            ydata = df[col].values.astype(float)

            # No aggressive cutoff for double peak
            y_fitdata = ydata
            wavelengths_fit = wavelengths

            # Initial guesses
            peak_index = np.argmax(ydata)
            peak_wavelength = wavelengths[peak_index]

            separation_guess = 1.0  # Ruby R-line separation ≈ 1 nm

            B_guess = (wavelengths.max() - wavelengths.min()) / 20

            p0 = [
                ydata.max()/2,              # A1
                peak_wavelength - 0.5,      # x01
                B_guess,                    # B1
                ydata.max()/2,              # A2
                peak_wavelength + 0.5,      # x02
                B_guess,                    # B2
                ydata.min()                 # C
            ]

            popt, cov = curve_fit(
                double_lorentzian,
                xdata=wavelengths_fit,
                ydata=y_fitdata,
                p0=p0,
                maxfev=20000
            )

            x01 = popt[1]
            x02 = popt[4]

            x01_list.append(x01)
            x02_list.append(x02)

            # ---------- INDIVIDUAL FIT PLOT ----------
            plt.figure(1)
            plt.scatter(wavelengths_fit, y_fitdata, color="black", s=3)
            plt.plot(
                wavelengths_fit,
                double_lorentzian(wavelengths_fit, *popt),
                color="grey"
            )

            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.minorticks_on()
            plt.tick_params(axis='both', which='major', size=6, direction='in')
            plt.tick_params(axis='both', which='minor', size=3, direction='in')
            plt.show()

            print(" ----- Results -----")
            print(f"x01: {x01}")
            print(f"x02: {x02}")
            print("----- ----- -----")

        # ---------- SHEET SUMMARY ----------
        mean_x01 = np.mean(x01_list)
        err_x01 = np.std(x01_list)/np.sqrt(len(x01_list))

        mean_x02 = np.mean(x02_list)
        err_x02 = np.std(x02_list)/np.sqrt(len(x02_list))

        print(f"\nMean x01: {mean_x01:.4f} +- {err_x01:.4f}")
        print(f"Mean x02: {mean_x02:.4f} +- {err_x02:.4f}")

        results[sheet] = {
            "mean_x01": mean_x01,
            "err_x01": err_x01,
            "mean_x02": mean_x02,
            "err_x02": err_x02,
            "x01_values": x01_list,
            "x02_values": x02_list
        }

    return results


#changes




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

def intensity_examiner(filename, sheet_name, column_name, start_index, cutoff = 0.1):

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


def double_lorentzian_intensity_extractor(
    filename,
    sheet_name,
    column_name,
    start_index,
    amplitude_frac=0.01,
    min_points=25,
    min_width=0.1,
    max_width=10
):

    results = {}

    # ---- Normalise sheet input ----
    if isinstance(sheet_name, str):
        sheet_name = [sheet_name]

    for s in range(len(sheet_name)):

        sheet = sheet_name[s]
        cols = column_name[s]
        start = start_index[s]

        print(f"\nProcessing sheet: {sheet}")

        df = pd.read_excel(filename, sheet_name=sheet).dropna()

        # ---------- WAVELENGTH EXTRACTION ----------
        indexes = np.arange(0, len(df[cols[0]]), 1)
        wavelengths = wavelength_extraction(indexes, start)
        wavelengths = np.array(wavelengths, dtype=float)

        x01_list = []
        x02_list = []

        # ---------- FIT EACH SPECTRUM ----------
        for col in cols:

            ydata = df[col].values.astype(float)

            # ---- Peak estimate ----
            peak_guess = np.argmax(ydata)
            Amax = ydata[peak_guess]

            # ---- Amplitude-based mask ----
            raw_mask = ydata > amplitude_frac * Amax
            valid_indices = np.where(raw_mask)[0]

            # ---- Keep contiguous region around peak ----
            splits = np.where(np.diff(valid_indices) > 1)[0]
            blocks = np.split(valid_indices, splits + 1)

            for block in blocks:
                if peak_guess in block:
                    fit_indices = block
                    break
            else:
                raise RuntimeError(f"No valid fit region found in sheet '{sheet}'")

            # ---- Minimum points check ----
            if len(fit_indices) < min_points:
                raise RuntimeError(
                    f"Too few points in fit region ({len(fit_indices)} < {min_points})"
                )

            wavelengths_fit = wavelengths[fit_indices]
            y_fitdata = ydata[fit_indices]

            # ---- Initial guesses ----
            peak_wavelength = wavelengths[peak_guess]
            separation_guess = 1.0

            B_guess = (wavelengths_fit.max() - wavelengths_fit.min()) / 5

            p0 = [
                Amax/2,                          # A1
                peak_wavelength - 0.5,            # x01
                B_guess,                          # B1
                Amax/2,                          # A2
                peak_wavelength + 0.5,            # x02
                B_guess,                          # B2
                y_fitdata.min()                  # C
            ]

            # ---- Bounds ----
            bounds = (
                [
                    0,
                    peak_wavelength - 2,
                    min_width,
                    0,
                    peak_wavelength - 2,
                    min_width,
                    -np.inf
                ],
                [
                    np.inf,
                    peak_wavelength + 2,
                    max_width,
                    np.inf,
                    peak_wavelength + 2,
                    max_width,
                    np.inf
                ]
            )

            # ---- Fit ----
            popt, cov = curve_fit(
                double_lorentzian,
                xdata=wavelengths_fit,
                ydata=y_fitdata,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            x01 = popt[1]
            x02 = popt[4]

            x01_list.append(x01)
            x02_list.append(x02)

            # ---------- Plot ----------
            x_dense = np.linspace(wavelengths_fit.min(),
                                  wavelengths_fit.max(), 2000)

            plt.figure()
            plt.scatter(wavelengths, ydata, s=4, color='grey', label='All data')
            plt.scatter(wavelengths_fit, y_fitdata, s=10,
                        color='black', label='Fit region')
            plt.plot(
                x_dense,
                double_lorentzian(x_dense, *popt),
                color='red',
                linestyle='--',
                label='Double Lorentzian fit'
            )
            plt.axvline(x01, color='blue', linestyle=':', label='x01')
            plt.axvline(x02, color='green', linestyle=':', label='x02')

            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.legend()
            plt.tight_layout()
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

        print(f"\nMean x01: {mean_x01:.4f} ± {err_x01:.4f}")
        print(f"Mean x02: {mean_x02:.4f} ± {err_x02:.4f}")

        results[sheet] = {
            "mean_x01": mean_x01,
            "err_x01": err_x01,
            "mean_x02": mean_x02,
            "err_x02": err_x02,
            "x01_values": x01_list,
            "x02_values": x02_list
        }

    return results








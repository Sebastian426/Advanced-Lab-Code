import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit




#functions we need to call inside the big functions

def chi_squared(model_params, model, x_data, y_data, y_error):
    return(np.sum(((y_data - model(x_data, *model_params))/y_error)**2))

def reduced_chi_squared(Chi_squared, DoF):
    return Chi_squared / (DoF)

def Lorentzian_Function(x, A, x0, B, C):
    return A / (1 + ((x - x0)/B)**2) + C


#Lorentzian fitter and peak index extractor

def peak_extractor(
    file,
    sheets,
    cols,
    amplitude_frac=0.01,
    uncertainty_frac=1e-3,
    min_points=15,
    min_width=5,
    max_width=300
):
    peak_indexes = []

    for sheet, sheet_cols in zip(sheets, cols):

        # ---- Load data ----
        df = pd.read_excel(file, sheet_name=sheet).dropna()
        amplitudes = [df[col].to_numpy() for col in sheet_cols]
        amplitudes = np.vstack(amplitudes)

        # ---- Bin & uncertainties ----
        mean_amp = amplitudes.mean(axis=0)
        std_amp  = amplitudes.std(axis=0, ddof=1)

        # ---- Uncertainty floor ----
        nonzero = std_amp[std_amp > 0]
        if len(nonzero) == 0:
            raise RuntimeError(f"All uncertainties zero in sheet '{sheet}'")

        floor = uncertainty_frac * np.max(nonzero)
        std_amp = np.maximum(std_amp, floor)

        index_list = np.arange(len(mean_amp))

        # ---- Peak estimate ----
        peak_guess = np.argmax(mean_amp)
        Amax = mean_amp[peak_guess]

        # ---- Amplitude-based mask ----
        raw_mask = mean_amp > amplitude_frac * Amax

        # ---- Keep contiguous region around peak ----
        valid_indices = np.where(raw_mask)[0]

        # Find block containing the peak
        splits = np.where(np.diff(valid_indices) > 1)[0]
        blocks = np.split(valid_indices, splits + 1)

        for block in blocks:
            if peak_guess in block:
                fit_indices = block
                break
        else:
            raise RuntimeError(f"No valid fit region found for sheet '{sheet}'")

        # ---- Enforce minimum size ----
        if len(fit_indices) < min_points:
            raise RuntimeError(
                f"Too few points in fit region for sheet '{sheet}' "
                f"({len(fit_indices)} < {min_points})"
            )

        x_fitdata = index_list[fit_indices]
        y_fitdata = mean_amp[fit_indices]
        err_fitdata = std_amp[fit_indices]

        # ---- Initial guess ----
        p0 = [
            y_fitdata.max(),             # A
            peak_guess,                  # x0
            len(fit_indices) / 4,         # B
            y_fitdata.min()               # C
        ]

        # ---- Bounds ----
        bounds = (
            [0, peak_guess - 5, min_width, -np.inf],
            [np.inf, peak_guess + 5, max_width, np.inf]
        )

        # ---- Fit ----
        popt, cov = curve_fit(
            Lorentzian_Function,
            xdata=x_fitdata,
            ydata=y_fitdata,
            sigma=err_fitdata,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds
        )

        # ---- Statistics ----
        dof = len(x_fitdata) - len(popt)
        chi2 = chi_squared(
            Lorentzian_Function,
            popt,
            x_fitdata,
            y_fitdata,
            err_fitdata
        )
        rchi2 = reduced_chi_squared(chi2, dof)

        peak = popt[1]
        peak_err = np.sqrt(cov[1, 1])
        peak_indexes.append(peak)

        # ---- Plot ----
        x_dense = np.linspace(x_fitdata.min(), x_fitdata.max(), 2000)

        plt.figure()
        plt.scatter(index_list, mean_amp, s=6, color='grey', label='All data')
        plt.scatter(x_fitdata, y_fitdata, s=12, color='black', label='Fit region')
        plt.plot(
            x_dense,
            Lorentzian_Function(x_dense, *popt),
            color='red',
            linestyle='--',
            label='Lorentzian fit'
        )
        plt.axvline(peak, color='blue', linestyle=':', label='Peak')

        plt.title(f'{sheet} Lorentzian Fit')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- Print results ----
        print(" ---- Results ---- ")
        print(f"Sheet: {sheet}")
        print(f"Peak index = {peak:.3f} ± {peak_err:.3f}")
        print(f"Width B = {popt[2]:.2f}")
        print(f"Reduced χ² = {rchi2:.3f}")
        print(f"Amplitude cutoff = {amplitude_frac:.1%} of A_max")
        print("----- ----- -----\n")

    return np.array(peak_indexes)


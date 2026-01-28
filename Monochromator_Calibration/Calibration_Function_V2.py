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

def peak_extractor(file, sheets, cols):
    peak_indexes = []

    for sheet, sheet_cols in zip(sheets, cols):

        # ---- Load data ----
        df = pd.read_excel(file, sheet_name=sheet).dropna()

        amplitudes = [df[col].to_numpy() for col in sheet_cols]
        amplitudes = np.vstack(amplitudes)

        # ---- Bin & uncertainties ----
        mean_amp = amplitudes.mean(axis=0)
        std_amp  = amplitudes.std(axis=0, ddof=1)

        index_list = np.arange(len(mean_amp))

        # ---- Initial guess ----
        p0 = [
            mean_amp.max(),                 # A
            np.argmax(mean_amp),            # x0
            10,                              # B (width)
            mean_amp.min()                  # C
        ]

        # ---- Fit ----
        popt, cov = curve_fit(
            Lorentzian_Function,
            xdata=index_list,
            ydata=mean_amp,
            sigma=std_amp,
            absolute_sigma=True,
            p0=p0
        )

        # ---- Statistics ----
        dof = len(mean_amp) - len(popt)
        chi2 = chi_squared(
            model=Lorentzian_Function,
            model_params=popt,
            x_data=index_list,
            y_data=mean_amp,
            y_error=std_amp
        )
        rchi2 = reduced_chi_squared(chi2, dof)

        peak = popt[1]
        peak_err = np.sqrt(cov[1, 1])
        peak_indexes.append(peak)

        # ---- Plot ----
        x_fit = np.linspace(index_list.min(), index_list.max(), 2000)

        plt.figure()
        plt.scatter(index_list, mean_amp, color='black', s=8, label='Binned data')
        plt.plot(x_fit,
                 Lorentzian_Function(x_fit, *popt),
                 color='red', linestyle='--', label='Lorentzian fit')
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
        print(f"Reduced χ² = {rchi2:.3f}")
        print("----- ----- -----\n")

    return np.array(peak_indexes)

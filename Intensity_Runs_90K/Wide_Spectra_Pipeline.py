import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14
})

# ============================================================
# Helper Functions
# ============================================================

def reduced_chi_squared(xdata, ydata, yerror, model, params):
    dof = len(ydata) - len(params)
    theoretical = model(xdata, *params)
    chi2 = np.sum(((ydata - theoretical) / yerror) ** 2)
    return chi2 / dof


def wavelength_extraction(x, start_index):
    A = 0.9958
    alpha_A = 0.0001

    B = 9096.6280
    alpha_B = 0.0289

    wavelengths = []
    uncertainties = []

    for i in range(len(x)):
        wavelength = (start_index - 0.242*x[i] - B) / A
        uncertainty = np.abs((x[i] - (B + alpha_B)) / (A + alpha_A) - wavelength)

        wavelengths.append(wavelength)
        uncertainties.append(uncertainty)

    return np.array(wavelengths), np.array(uncertainties)


def voigt_model_double(x, A1, center1, gamma1, sigma,
                       A2, center2, gamma2, offset):

    return (
        A1 * voigt_profile(x - center1, sigma, gamma1) +
        A2 * voigt_profile(x - center2, sigma, gamma2) +
        offset
    )


# ============================================================
# Pipeline Steps
# ============================================================

def load_data(file, sheet, cols):
    df = pd.read_excel(file, sheet_name=sheet)
    index = np.arange(0, len(df[cols[0]]), 1)
    return df, index


def normalise_spectra(df, cols, wavelengths):
    dx = np.abs(wavelengths[1] - wavelengths[0])

    area_0 = np.sum(df[cols[0]] * dx)
    normalised_spectra = []

    for col in cols:
        spectrum = df[col].to_numpy()
        norm_spec = (spectrum / np.sum(spectrum * dx)) * area_0
        normalised_spectra.append(norm_spec)

    intensity_matrix = np.column_stack(normalised_spectra)

    intensity_data = np.mean(intensity_matrix, axis=1)
    intensity_error = np.std(intensity_matrix, axis=1) / np.sqrt(len(cols))

    return normalised_spectra, intensity_data, intensity_error


def fit_double_voigt(wavelengths, intensity_data, intensity_error, padding):

    center2_guess = wavelengths[np.argmax(intensity_data)]
    center1_guess = center2_guess - 1.4

    p0 = [
        intensity_data.max()/2,
        center1_guess,
        0.3,
        0.3,
        intensity_data.max()/2,
        center2_guess,
        0.3,
        np.min(intensity_data)
    ]

    bounds = [[
        0,
        wavelengths[padding[1]],
        0,
        0,
        0,
        wavelengths[padding[1]],
        0,
        -np.inf
    ],[
        np.inf,
        wavelengths[padding[0]],
        np.inf,
        np.inf,
        np.inf,
        wavelengths[padding[0]],
        np.inf,
        np.inf
    ]]

    pop, cov = curve_fit(
        voigt_model_double,
        wavelengths[padding[0]:padding[1]],
        intensity_data[padding[0]:padding[1]],
        sigma=intensity_error[padding[0]:padding[1]],
        absolute_sigma=True,
        p0=p0,
        bounds=bounds
    )

    residuals = (
        intensity_data[padding[0]:padding[1]] -
        voigt_model_double(wavelengths[padding[0]:padding[1]], *pop)
    ) / intensity_error[padding[0]:padding[1]]

    rchi2 = reduced_chi_squared(
        wavelengths[padding[0]:padding[1]],
        intensity_data[padding[0]:padding[1]],
        intensity_error[padding[0]:padding[1]],
        voigt_model_double,
        pop
    )

    return pop, cov, residuals, rchi2


# ============================================================
# Plotting
# ============================================================

def plot_raw(wavelengths, df, cols):

    plt.figure()

    for col in cols:
        plt.plot(wavelengths, df[col], label=col)

    plt.title("Raw Spectra")
    plt.ylabel("Intensity")
    plt.xlabel("Wavelength (nm)")

    plt.minorticks_on()
    plt.tick_params(which='major', direction='in',
                    top=True, bottom=True, left=True, right=True, size=6)
    plt.tick_params(which='minor', direction='in',
                    top=True, bottom=True, left=True, right=True, size=3)

    plt.show()


def plot_normalised(wavelengths, normalised_spectra, padding):

    plt.figure()

    for spec in normalised_spectra:
        plt.plot(wavelengths, spec, alpha=0.7)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalised Intensity")
    plt.title("Normalised Spectra")

    plt.axvline(wavelengths[padding[0]], color='green')
    plt.axvline(wavelengths[padding[1]], color='red')

    plt.minorticks_on()
    plt.tick_params(which='major', direction='in',
                    top=True, bottom=True, left=True, right=True, size=6)
    plt.tick_params(which='minor', direction='in',
                    top=True, bottom=True, left=True, right=True, size=3)

    plt.show()


def plot_fit(wavelengths, intensity_data, intensity_error,
             padding, pop, residuals):

    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_axes([0,0,1,1])

    ax1.scatter(wavelengths, intensity_data, c='black', s=5)

    ax1.errorbar(wavelengths, intensity_data,
                 yerr=intensity_error,
                 linestyle="none",
                 color="black",
                 capsize=3)

    ax1.plot(wavelengths,
             voigt_model_double(wavelengths, *pop),
             color="grey",
             linestyle="--")

    ax1.set_ylabel("Intensity")

    ax1.minorticks_on()
    ax1.tick_params(which='major', direction='in',
                    top=True, bottom=True, left=True, right=True, size=6)
    ax1.tick_params(which='minor', direction='in',
                    top=True, bottom=True, left=True, right=True, size=3)

    x1 = wavelengths[padding[0]]
    x2 = wavelengths[padding[1]]

    ax1.axvspan(x1, x2, alpha=0.1, color='grey')

    ax2 = inset_axes(ax1,
                     width="100%",
                     height="30%",
                     loc="lower left",
                     bbox_to_anchor=(0, -0.35, 1, 1),
                     bbox_transform=ax1.transAxes)

    ax2.scatter(wavelengths[padding[0]:padding[1]],
                residuals,
                s=5,
                color="black")

    ax2.axhspan(-np.std(residuals),
                np.std(residuals),
                color="grey",
                alpha=0.25)

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Normalised Residuals")

    ax2.minorticks_on()
    ax2.tick_params(which='major', direction='in',
                    top=True, bottom=True, left=True, right=True, size=6)
    ax2.tick_params(which='minor', direction='in',
                    top=True, bottom=True, left=True, right=True, size=3)

    plt.show()


# ============================================================
# Summary Output
# ============================================================

def print_summary(pop, cov, rchi2):

    print("------ Summary Statistics ------")
    print(f"Reduced Chi Squared: {rchi2:.4f}")

    if pop[0] > pop[4]:
        R1, R2 = 1, 5
    else:
        R1, R2 = 5, 1

    print("----- R1 -----")
    print("R1 Theoretical: 694.3")
    print(f"R1 Observed: {pop[R1]:.4f} ± {np.sqrt(cov[R1][R1])}")

    print("----- R2 -----")
    print("R2 Theoretical: 692.9")
    print(f"R2 Observed: {pop[R2]:.4f} ± {np.sqrt(cov[R2][R2])}")


# ============================================================
# Master Pipeline
# ============================================================

def run_pipeline(file, sheet, cols, start_index, padding):

    df, index = load_data(file, sheet, cols)

    wavelengths, _ = wavelength_extraction(index, start_index)

    plot_raw(wavelengths, df, cols)

    normalised_spectra, intensity_data, intensity_error = normalise_spectra(
        df, cols, wavelengths
    )

    plot_normalised(wavelengths, normalised_spectra, padding)

    pop, cov, residuals, rchi2 = fit_double_voigt(
        wavelengths,
        intensity_data,
        intensity_error,
        padding
    )

    plot_fit(wavelengths,
             intensity_data,
             intensity_error,
             padding,
             pop,
             residuals)

    print_summary(pop, cov, rchi2)

    return pop, cov
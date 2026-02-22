import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "font.size": 14,          # Base font size
    "axes.titlesize": 14,     # Title
    "axes.labelsize": 14,     # Axis labels
    "xtick.labelsize": 12,    # X ticks
    "ytick.labelsize": 12,    # Y ticks
    "legend.fontsize": 12,    # Legend
    "figure.titlesize": 14
})

# ============================================================
# Models
# ============================================================

def voigt_model_double(x, A1, center1, gamma1, sigma,
                       A2, center2, gamma2, offset):

    return (
        A1 * voigt_profile(x - center1, sigma, gamma1) +
        A2 * voigt_profile(x - center2, sigma, gamma2) +
        offset
    )


def reduced_chi_squared(xdata, ydata, yerror, model, params):
    dof = len(ydata) - len(params)
    theory = model(xdata, *params)
    return np.sum(((ydata - theory) / yerror) ** 2) / dof


def wavelength_extraction(x, start_index):
    A = 0.9958
    B = 9096.6280
    return (start_index - 0.242*x - B) / A

def load_data(file, sheet, cols):
    df = pd.read_excel(file, sheet_name=sheet)
    index = np.arange(len(df[cols[0]]))
    return df, index


def normalise_spectra(df, cols, wavelengths):

    dx = np.abs(wavelengths[1] - wavelengths[0])
    area_0 = np.sum(df[cols[0]] * dx)

    normalised = []

    for col in cols:
        spec = df[col].to_numpy()
        norm = (spec / np.sum(spec * dx)) * area_0
        normalised.append(norm)

    matrix = np.column_stack(normalised)

    mean_intensity = np.mean(matrix, axis=1)
    error = np.std(matrix, axis=1) / np.sqrt(len(cols))

    return normalised, mean_intensity, error

def fit_voigt_temperature(wavelengths, intensity, error, padding):

    center2_guess = wavelengths[np.argmax(intensity)]
    center1_guess = center2_guess - 1.4

    p0 = [
        intensity.max()/2,
        center1_guess,
        0.3,
        0.3,
        intensity.max()/2,
        center2_guess,
        0.3,
        np.min(intensity)
    ]

    pop, cov = curve_fit(
        voigt_model_double,
        wavelengths[padding[0]:padding[1]],
        intensity[padding[0]:padding[1]],
        sigma=error[padding[0]:padding[1]],
        absolute_sigma=True,
        p0=p0
    )

    residuals = (
        intensity[padding[0]:padding[1]] -
        voigt_model_double(wavelengths[padding[0]:padding[1]], *pop)
    ) / error[padding[0]:padding[1]]

    rchi2 = reduced_chi_squared(
        wavelengths[padding[0]:padding[1]],
        intensity[padding[0]:padding[1]],
        error[padding[0]:padding[1]],
        voigt_model_double,
        pop
    )

    return pop, cov, residuals, rchi2

def plot_fit(wavelengths, intensity, error, pop, temp,padding):

    plt.figure()

    A1, center1, gamma1, sigma, A2, center2, gamma2, offset = pop

    plt.errorbar(wavelengths,
                 intensity,
                 yerr=error,
                 fmt='o',
                 ms=3,
                 label="Data",
                 color = "black",
                 capsize=3)

    plt.plot(wavelengths,
             voigt_model_double(wavelengths, *pop),
             label="Voigt Fit",
             color = "grey",
             linestyle = "--")
    plt.axvline(wavelengths[padding[0]], color = "red")
    plt.axvline(wavelengths[padding[1]], color = "red")

    plt.axvline(center1, color = "blue")
    plt.axvline(center2, color = "blue")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"{temp} K")

    plt.legend()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', size=6, direction='in',
                    right=True, left=True, top=True, bottom=True)
    plt.tick_params(axis='both', which='minor', size=3, direction='in',
                    right=True, left=True, top=True, bottom=True)
    plt.show()

def run_pipeline_temperature(file, sheet, cols,
                             start_index,
                             padding,
                             plot=False):

    df, index = load_data(file, sheet, cols)

    wavelengths = wavelength_extraction(index, start_index)

    normalised, intensity, error = normalise_spectra(
        df, cols, wavelengths
    )

    pop, cov, residuals, rchi2 = fit_voigt_temperature(
        wavelengths,
        intensity,
        error,
        padding
    )

    if plot:
        plot_fit(wavelengths, intensity, error, pop, sheet, padding)

    return wavelengths, intensity, error, pop, cov, rchi2

def run_pipeline_all_temperatures(file,
                                  sheets,
                                  cols,
                                  start_index,
                                  padding,
                                  plot=False):

    results = []
    all_intensities = {}
    wavelengths_ref = None

    for sheet, columns in zip(sheets, cols):

        wavelengths, intensity, error, pop, cov, rchi2 = (
            run_pipeline_temperature(
                file,
                sheet,
                columns,
                start_index,
                padding,
                plot
            )
        )

        # Store spectra for combined plot
        all_intensities[sheet] = intensity
        if wavelengths_ref is None:
            wavelengths_ref = wavelengths

        results.append({
            "Temperature": float(sheet),
            "Center1": pop[1],
            "Center1_err": np.sqrt(cov[1][1]),
            "Center2": pop[5],
            "Center2_err": np.sqrt(cov[5][5]),
            "Sigma": pop[3],
            "Sigma_err": np.sqrt(cov[3][3]),
            "Gamma1": pop[2],
            "Gamma2": pop[6],
            "Reduced_Chi2": rchi2
        })

    # Plot all spectra together
    plt.figure()

    for sheet in sheets:
        plt.plot(wavelengths_ref, all_intensities[sheet],
                 label=f"{sheet} K")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("All Temperatures")
    plt.legend()

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', size=6, direction='in',
                    right=True, left=True, top=True, bottom=True)
    plt.tick_params(axis='both', which='minor', size=3, direction='in',
                    right=True, left=True, top=True, bottom=True)

    plt.show()

    summary_table = pd.DataFrame(results)

    return wavelengths_ref, summary_table
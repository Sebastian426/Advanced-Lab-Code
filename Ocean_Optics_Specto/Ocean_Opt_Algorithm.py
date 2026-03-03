import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.integrate import quad

Td_fixed = 760

def voigt_model_double(x, A1, center1, gamma1, sigma1,
                       A2, delta, gamma2, sigma2, offset):

    return (
        A1 * voigt_profile(x - center1, sigma1, gamma1) +
        A2 * voigt_profile(x - (center1 + delta), sigma2, gamma2) +
        offset
    )

def voigt_model_single(x,A1, centre1, gamma1, sigma1):
    return(A1*voigt_profile(x - centre1, sigma1, gamma1))

def reduced_chi_squared(x, y, model, pop):
    theory = model(x,*pop)
    dof = len(x)-len(pop)
    chi2 = np.sum((y-theory)**2/theory)
    redchi2 = chi2/dof
    return redchi2

def debye_integrand(x):
    return x**3 / (np.exp(x) - 1)

def debye_integral(T, Td):

    if T <= 0:
        return 0  # avoid divide by zero

    upper = min(Td/T, 50)

    result, _ = quad(
        lambda x: x**3/(np.exp(x)-1),
        0,
        upper
    )

    return result

def ruby_shift_model_fixed(T, R0, alpha):
    integral_vals = np.array([debye_integral(t, Td_fixed) for t in T])
    return R0 + alpha * (T / Td_fixed)**4 * integral_vals


def analyze_ruby_spectra(filename, column_names,
                         wavelength_min=688,
                         wavelength_max=700,
                         center1_guesses=None,
                         Td_fixed_input=None,
                         p0_estimate=None,
                         bounds_estimate=None):
    global Td_fixed
    Td_fixed = Td_fixed_input

    df = pd.read_excel(filename)
    results = {}

    for cols in column_names:
        w_col, i_col = cols
        temperature = int(w_col.split("_")[-1])

        # Load data
        wavelengths_raw = df[w_col].to_numpy()
        intensity_raw = df[i_col].to_numpy()

        # Remove NaNs
        valid = ~np.isnan(wavelengths_raw) & ~np.isnan(intensity_raw)
        wavelengths_raw = wavelengths_raw[valid]
        intensity_raw = intensity_raw[valid]

        # Filter wavelength range
        mask = (wavelengths_raw > wavelength_min) & (wavelengths_raw < wavelength_max)
        wavelengths = wavelengths_raw[mask]
        intensities = intensity_raw[mask]

        if len(wavelengths) < 10:
            print(f"Not enough data at {temperature}K")
            continue

        # Choose center guess
        if center1_guesses is not None and temperature in center1_guesses:
            center1_guess = center1_guesses[temperature]
        else:
            center1_guess = 692.9

        # Initial guesses
        if p0_estimate is None:
            p0 = [
                np.max(intensities) / 2,
                center1_guess,
                0.3,
                0.3,
                np.max(intensities) / 2,
                1.3,
                0.3,
                0.3,
                np.min(intensities)
            ]
        else:
            p0 = p0_estimate

        # Bounds
        if bounds_estimate is None:
            bounds = (
                [0, 691, 0, 0, 0, 0.5, 0, 0, 0],
                [np.inf, 693.5, np.inf, np.inf, np.inf, 4, np.inf, np.inf, np.inf]
            )
        else:
            bounds = bounds_estimate

        # Fit
        try:
            pop, cov = curve_fit(
                voigt_model_double,
                wavelengths,
                intensities,
                p0=p0,
                bounds=bounds,
                maxfev=30000
            )
        except RuntimeError:
            print(f"Fit failed at {temperature}K")
            continue

        rchi2 = reduced_chi_squared(wavelengths, intensities, voigt_model_double, pop)

        # Split peaks
        pop_r1 = np.array([pop[0], pop[1], pop[2], pop[3]])
        pop_r2 = np.array([pop[4], pop[1] + pop[5], pop[6], pop[7]])

        # Plot
        plt.figure()
        plt.scatter(wavelengths, intensities, s=5, c="black")

        plt.plot(wavelengths,
                 voigt_model_double(wavelengths, *pop),
                 linestyle="--",
                 color="grey")

        plt.plot(wavelengths,
                 voigt_model_single(wavelengths, *pop_r1) + pop[-1],
                 c="b",
                 linestyle="--",
                 alpha=0.5)

        plt.plot(wavelengths,
                 voigt_model_single(wavelengths, *pop_r2) + pop[-1],
                 c="r",
                 linestyle="--",
                 alpha=0.5)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")

        plt.axvline(pop[1] + pop[5], color="r")
        plt.axvline(pop[1], color="b")

        plt.title(f"Spectrum at {temperature} K")
        plt.show()

        # Uncertainties
        diag_cov = np.diag(cov)

        r1_obs = pop[1] + pop[5]
        r1_err = np.sqrt(diag_cov[1] + diag_cov[5])
        r2_obs = pop[1]
        r2_err = np.sqrt(diag_cov[1])

        r1_frac = pop[4] / (pop[0] + pop[4])
        r2_frac = pop[0] / (pop[0] + pop[4])

        print(f"----- Results ({temperature} K) -----")
        print(f"R1 Theory: 693.4")
        print(f"R1 Obs: {r1_obs} ± {r1_err}")
        print(f"R1 frac: {r1_frac}")
        print(f"R2 Theory: 692.9")
        print(f"R2 Obs: {r2_obs} ± {r2_err}")
        print(f"R2 frac: {r2_frac}")
        print(f"Reduced Chi Squared: {rchi2}")

        results[temperature] = {
            "R1_obs": r1_obs,
            "R1_err": r1_err,
            "R1_frac": r1_frac,
            "R2_obs": r2_obs,
            "R2_err": r2_err,
            "R2_frac": r2_frac,
            "reduced_chi2": rchi2,
            "parameters": pop,
            "covariance": cov
        }

    # ============================
    # Temperature shift fitting
    # ============================
    if Td_fixed_input is not None:

        # Automatically detect temperatures
        temperatures = sorted(results.keys())
        temperatures = np.array(temperatures)

        R1_vals = []
        R1_errs = []
        R2_vals = []
        R2_errs = []

        for temp in temperatures:
            R1_vals.append(results[temp]["R1_obs"])
            R1_errs.append(results[temp]["R1_err"])
            R2_vals.append(results[temp]["R2_obs"])
            R2_errs.append(results[temp]["R2_err"])

        R1_vals = np.array(R1_vals)
        R1_errs = np.array(R1_errs)
        R2_vals = np.array(R2_vals)
        R2_errs = np.array(R2_errs)

        # Auto fit range
        T_fit = np.linspace(np.min(temperatures),
                            np.max(temperatures),
                            400)

        # Fit R1
        popt_R1, pcov_R1 = curve_fit(
            ruby_shift_model_fixed,
            temperatures,
            R1_vals,
            p0=[R1_vals[0], 1]
        )

        # Fit R2
        popt_R2, pcov_R2 = curve_fit(
            ruby_shift_model_fixed,
            temperatures,
            R2_vals,
            p0=[R2_vals[0], 1]
        )

        # Plot R1
        plt.figure()
        plt.errorbar(temperatures, R1_vals,
                     yerr=R1_errs,
                     fmt='o',
                     label="Data")

        plt.plot(T_fit,
                 ruby_shift_model_fixed(T_fit, *popt_R1),
                 '--',
                 label="Fit")

        plt.xlabel("Temperature (K)")
        plt.ylabel("Peak Position (nm)")
        plt.title("Shift of R1 Peak")
        plt.legend()
        plt.show()

        # Plot R2
        plt.figure()
        plt.errorbar(temperatures, R2_vals,
                     yerr=R2_errs,
                     fmt='o',
                     label="Data")

        plt.plot(T_fit,
                 ruby_shift_model_fixed(T_fit, *popt_R2),
                 '--',
                 label="Fit")

        plt.xlabel("Temperature (K)")
        plt.ylabel("Peak Position (nm)")
        plt.title("Shift of R2 Peak")
        plt.legend()
        plt.show()

        # Store fit results
        results["temperature_fit"] = {
            "Td_fixed": Td_fixed,
            "temperatures": temperatures,
            "R1_vals": R1_vals,
            "R2_vals": R2_vals,
            "R1_errs": R1_errs,
            "R2_errs": R2_errs,
            "R1_params": popt_R1,
            "R2_params": popt_R2,
            "R1_cov": pcov_R1,
            "R2_cov": pcov_R2
        }

    return results
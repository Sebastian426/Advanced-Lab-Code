import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

def residual(theoretical, observed):
    return theoretical - observed

def normalised_residual(theoretical, observed, error):
    res = theoretical - observed / error
    return  res/np.std(res)

def chi_squared(model_params, model, x_data, y_data, y_error):
    return(np.sum(((y_data - model(x_data, *model_params))/y_error)**2))

def reduced_chi_squared(Chi_squared, DoF):
    return Chi_squared / (DoF - 1)

def DataLoader(file_name, independent_variable_name, dependent_variable_name, independent_variable_uncertainty_name, dependent_variable_uncertainty_name, model, p0_model):

    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_name)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_name)
    else:
        raise ValueError("Unsupported file type")

    x_data = df[independent_variable_name].to_numpy()
    y_data = df[dependent_variable_name].to_numpy()

    x_error = df[independent_variable_uncertainty_name].to_numpy()
    y_error = df[dependent_variable_uncertainty_name].to_numpy()
    y_error = np.abs(y_error)

    dof = len(x_data) - len(p0_model)

    if len(x_data) <= len(p0_model):
        raise ValueError("Not enough data points for number of fit parameters")

    if dof <= 0:
        raise ValueError("Degrees of freedom <= 0. Not enough data points.")


    popt_model, cov_model = curve_fit(model,
                                  x_data,
                                  y_data,
                                  sigma = y_error,
                                  p0 = p0_model,
                                  absolute_sigma = True) #if the error is accurate set to True, if guess set False

    model_uncertainties = np.sqrt(np.diag(cov_model))
    chi2 = chi_squared(popt_model, model, x_data, y_data, y_error)

    Rchi2 = reduced_chi_squared(chi2, dof)

    norm_res = normalised_residual(model(x_data, *popt_model), y_data, y_error)

    fig = plt.figure(1)

    main = fig.add_axes([0,0,1,1])
    main.errorbar(x_data, y_data, yerr = y_error, fmt='none', linestyle='None', color = 'black', label = 'Data')
    main.plot(x_data, model(x_data, *popt_model), label = "Model", color = 'grey', alpha = 0.5)


    plt.ylabel(dependent_variable_name)
    main.legend()

    resplt = fig.add_axes([0,-0.2,1,0.2])
    resplt.axhline(0, color='black')
    resplt.axhline(1, color='grey', linestyle = '--')
    resplt.axhline(-1, color='grey', linestyle = '--')

    resplt.scatter(x_data, norm_res, marker = 'D', color = 'black', s = 3)

    plt.xlabel(independent_variable_name)
    plt.ylabel("Normalised \n Residuals")
    plt.show()

    print(f"----- Fit Results -----")


    for i in range(len(popt_model)):
        print(f"fit variable {i+1} = {popt_model[i]:.5f}±{model_uncertainties[i]:.5f}")

    print(f"----- Data Analysis -----")
    print(f"Chi^2 = {chi2:.5f}")
    print(f"Reduced Chi^2 = {Rchi2:.5f}")
    print(f"----- ----- ----- -----")

    return{
    "x": x_data,
    "x_err": x_error,
    "y": y_data,
    "y_err": y_error,
    "popt": popt_model,
    "perr": model_uncertainties,
    "norm_res": norm_res,
    "chi2": chi2,
    "red_chi2": Rchi2}
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
INSTRUCTIONS
~~~~~~~~~~~~

This is a template file for you to use for your Computing 2 Coursework.

Save this file as py12spqr.py when py12spqr should be replaced with your ISS username

Do not rename or remove the function ProcessData, the marking will assume that this function exists
and will take in the name of a data file to open and process. It will assume that the function will return
all the answers your code finds. The results are returned as a dictionary.

Your code should also produce the same plot as you produce in your report.

This dicstring does not need to be included.

@author: phygbu
"""
# Your imports go here
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import mu_0, hbar,  physical_constants
from scipy.integrate import cumtrapz, simps

# Your functions go here
def find_decimal_place(number):
    '''finds the decimal place number to round an uncertainty to, so finds the position of the first significant figure
    in number. This is done by separating number into a list of strings, then if there is a decimal
    it finds the position of the first significant figure after the decimal, if there is a power it finds the value of
    the power. number is a float, returns an integer.'''
    list_int=list(str(number))
    decimal_place= len(list_int) #if error is 0.00000 then decimal place is length of number
    if list_int[0]=="0": #so if the number is 0.xxx then we start searching after the decimal point, and we know we
        #will need to use the positive decimal place number that has the first non zero value
        i=0
        for character in list_int[2:]:
            if character != "0":
                decimal_place=i+1
                break
            i+=1
    elif "e" in list_int:#for any value with 7 or more zero e.g. 0.000000x or x0000000, is written as e n where n is
        #the number of zeros- so if e is to the power of 7 this means we want to round to 7 places to the left
        i = list_int.index("e")#the index of e so that we can acess the numbers after e
        decimal_place = -int("".join(list_int[i+1:]))
    else: #so if the number is something like 573.456, then we start searching from the begining till we find the
        #decimal place, then the decimal to round to is -3 in this case.
        i=0
        for character in list_int:
            if character ==".":
                decimal_place = -(i-1)
                break
            i+=1
    return decimal_place

def lorentzian(H, deltaH, I_0, H_0):
    '''the lorenzian function'''
    bracket1 = (H-H_0)**2
    bracket2 = (deltaH/2)**2
    denominator = 2*np.pi*(bracket1 + bracket2)
    return ((I_0*deltaH)/denominator)

def differential_lorentzian(H,deltaH, I_0, H_0):
    '''the differential of the lorenzian function with respect to the magnetic field H'''
    prefactor = -(I_0*deltaH)/(2*np.pi)
    numerator = 2*(H-H_0)
    denominator = ((H-H_0)**2)+((deltaH/2)**2)
    return prefactor*numerator*(denominator**(-2))

def fit_differential_lorentzian(magnetic_field, differential_absorption):
    '''uses curve fitting to guess the parameters deltaH (the width of the peak), I_0 and H_0, the value of the magnetic
    field at the peak intensity. Curve fitting uses the data for magnetic field and differential absorption.'''
    max_x=np.argmax(cumtrapz(differential_absorption,magnetic_field, initial = 0)) #index of the maximum value measured 
    #for absorption. Using differential absorption means it first needs to be integrated to find the max in absorption
    H_0_guess = magnetic_field[max_x] #this is the value of the magneti field that gives the maximum absorption
    
    half_absorption = differential_absorption[:max_x] #knowing the shape of the graph, we use only half the differential data
    #i.e. only data up to max so all the data is positive, then the width of the peak is twize the width of this half peak
    average_absorption = (sum(half_absorption)) / half_absorption.shape[0]
    i=0
    for item in half_absorption:
        if item>average_absorption: #as soon as the value of absorption rises above the average, we use this as the base of our peak
            break
        i+=1 #i is then the index of the base of the peak
    deltaH_guess = ( H_0_guess- magnetic_field[i])/2
    
    I_0_guess = 2*simps(cumtrapz(half_absorption,magnetic_field[:max_x], initial = 0), magnetic_field[:max_x])
    #I_0_guess is the area under the peak, so twice the area under half the peak
    parameters, var_covar = curve_fit(differential_lorentzian,magnetic_field, differential_absorption, p0=(deltaH_guess, I_0_guess, H_0_guess),maxfev = 600000000)
    return parameters[0], parameters[1], parameters[2], var_covar


def Kittel(magnetic_field, gamma, H_k, M_s):
    '''the Kittel function, which gives the freuqnecy as a function of peak magnetic field'''
    prefactor = (mu_0*gamma)/(2*np.pi)
    square_root = (magnetic_field+H_k)*(magnetic_field+H_k+M_s)
    return prefactor * np.sqrt(square_root)

def fit_kittle_func(magnetic_field, absorption):
    '''returns the value of the gyromagnetic ratio gamma, the anistrophy field H_k and the saturation 
    magnetisation M_s using curve fitting to fit the kittel function with the frequency and peak magnetic field.'''
    mu_b = physical_constants["Bohr magneton"][0]
    parameters, var_covar = curve_fit(Kittel, magnetic_field, absorption, p0 = ((1.75*mu_b)/hbar, 20000,60000),  maxfev = 600000000)
    return parameters, var_covar


def linear_peak_width(frequency, deltaH_0, gyromagnetic_ratio):
    '''The linear relation between frequncy and the peak width'''
    top = gyromagnetic_ratio*4*np.pi*frequency
    bottom = mu_0*np.sqrt(3)
    return deltaH_0+(top/bottom)

def fit_peak_width(frequency, peak_width):
    '''returns the parameters deltaH_0 and the gyromagnetic ratio alpha/gamma, using the independent variable
    frequncy and the corresponding dependent variable the peak width'''
    parameters, var_covar = curve_fit(linear_peak_width, frequency, peak_width, p0 = (1, 0.05),  maxfev = 600000000)
    return parameters, var_covar
    

def results_for_all_frequ(magnetic_field, differential_absorption_data, frequencies):
    '''magnetic field is the independent variable, the absorption data is the dependent variable, and each row is the absorption
    data of each magnetic field value measured for the different frequncies in frequncies''' 
    results=np.zeros((4,len(differential_absorption_data))) #an 4 by 136 array, where the first row will be the 135 frequencies
    #and the second row will be the corresponding deltaH, the third row the corresponding I_0 and the fourth the corresponging H_0
    results_error = np.zeros((3,len(differential_absorption_data)))#an array of same shape as results, where each element is the systematic error
    #of the element in results with the same index.
    i=0
    for absorption in differential_absorption_data:
        frequ=frequencies[i+1].strip("GHz")
        width, I_0, peak, var_covar= fit_differential_lorentzian(magnetic_field, absorption)
        results[:,i]=np.array((frequ,width, I_0, peak))#this is column i in the results array
        results_error[:,i] = np.sqrt(np.diag(var_covar)) #The corresponding colum i in the error array
        i+=1
    return results, results_error

def graph_AbsorptionVMagneticField(magnetic_field, absorption, deltaH, I_0, H_0, deltaH_err, I_0_err, H_0_err ):
    '''this plots the absortiopn againsts the magnetic field for a certain frequncy if the results for the
   that frequency are the arrays magneticfield and absorption. deltaH, I_0, H_0 are the values found from fitting
   the curve to the lorenzian function, along with their corresponding errors.'''
    
    deltaH, deltaH_err = np.around([deltaH, deltaH_err], decimals = find_decimal_place(deltaH_err))
    I_0, I_0_err = np.around([I_0, I_0_err], decimals = find_decimal_place(I_0_err))
    H_0, H_0_err = np.around([H_0, H_0_err], decimals = find_decimal_place(H_0_err))
    
    plt.title('Absorption Vs Magnetic Field ($\u03BD = 20GHz$)')
    plt.ylabel('Absorption (arb units)')
    plt.xlabel('Magnetic Field $Am^-1$')
    plt.plot(magnetic_field, absorption, ".",label="data")
    Y_Values = lorentzian(magnetic_field, deltaH, I_0, H_0) #The ideal data that would have been collected 
    plt.plot(magnetic_field,Y_Values, label="fit" )
    plt.legend()
    y_peak = Y_Values[np.argmax(Y_Values)] #this is the amount of aborption that occurs at the peak for
    #the fitted data, which is used to locate where the arrow should point to
    plt.annotate(f"$I_0 = {I_0}±{I_0_err}$\n$H_0 = {H_0}±{H_0_err}$\n$∆H = {deltaH}±{deltaH_err}$",
            xy=(H_0, y_peak), xycoords='data',
            xytext=(magnetic_field[-1]*0.5, y_peak*0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )#xytext is the coordinated of the text, which are positioned halfway down the y axis and halfway along x axis
    plt.show()


def graph_FrequencyVPeakPosition(results, error_peak):
    '''plots the frequncy against the position in the magnetic field that has the maximum absorption for that frequency.
    results is the array that gives the data for all the frequncies. Returns the values for gamma, H_k and M_s calculated
    by fitting the data to the kittle function, and their corresponding errors. Also returns g_factor calculated from gamma
    and its corresponding (propagated) error'''

    parameters, var_covar = fit_kittle_func(results[3,:], results[0,:]*(10**9))
    gamma, H_k, M_s = parameters
    error_gammaH_kM_s = np.sqrt(np.diag(var_covar))
    mu_b = physical_constants["Bohr magneton"][0]
    g_factor = (gamma*hbar)/mu_b
    g_error = (error_gammaH_kM_s[0]*hbar)/mu_b

    plt.title('Frequency Vs peak position')
    plt.ylabel('Frequency (GHz)')
    plt.xlabel('Magnetic Field $Am^-1$')
    gradient = np.polyfit(results[3,:],results[0,:], 1 )[0]
    plt.errorbar(results[3,:], results[0,:], xerr=error_peak*gradient, fmt='.',label="data" )
    plt.plot(results[3,:],Kittel(results[3,:], gamma, H_k, M_s)/(10**9), label="fit")
    plt.legend()

    g_factor_rounded, g_error_rounded = np.around([g_factor, g_error], decimals = find_decimal_place((error_gammaH_kM_s[0]*hbar)/mu_b))
    H_k_rounded, H_error_rounded = np.around([H_k, error_gammaH_kM_s[1]], decimals = find_decimal_place(error_gammaH_kM_s[1]))
    M_s_rounded, M_s_error_rounded = np.around([M_s, error_gammaH_kM_s[2]], decimals = find_decimal_place(error_gammaH_kM_s[2]))
    plt.annotate(f"$g = {g_factor_rounded}±{g_error_rounded}$\n$H_k = {H_k_rounded}±{H_error_rounded}$\n$M_s = {M_s_rounded}±{M_s_error_rounded}$",
                xy=(0, results[0,-1]*0.5), xycoords='data',
                xytext=(0, results[0,-1]*0.5), textcoords='data',
                )
    plt.show()
    return g_factor, g_error, H_k, error_gammaH_kM_s[1], M_s, error_gammaH_kM_s[2], gamma, error_gammaH_kM_s[0]


def graph_PeakWidthVFrequency(results,error_peakWidth, gamma, gamma_error ):
    '''plots the width of the peak against the frequncy, where frequncy and results are both contained inside the results 
    array. the parameters of deltaH_0 and the gyromagnetic ratio are calculated by fitting the data to the 
    equation for finding peak width. Then alpha is found using the gyromagnetic ratio and gamma.
    Returns deltaH_0, alpha and their correspondng errors.'''
    plt.title("Peak width versus frequency")
    plt.ylabel('Peak Width $Am^-1$')
    plt.xlabel('Frequency (GHz)')
    plt.errorbar(results[0,:], results[1,:], yerr = error_peakWidth, fmt = ".", label="data")
    [deltaH_0, gyromagnetic_ratio], var_covar = fit_peak_width(results[0,:]*(10**9), results[1,:])
    deltaH_0_error, gyromagnetic_ratio_error = np.sqrt(np.diag(var_covar))
    alpha = gyromagnetic_ratio*gamma
    alpha_error = ((gyromagnetic_ratio_error/gyromagnetic_ratio)+(gamma_error/gamma))*alpha
    deltaH_0_rounded, deltaH_0_error_rounded = np.around([deltaH_0, deltaH_0_error], decimals = find_decimal_place(deltaH_0_error))
    alpha_rounded, alpha_error_rounded = np.around([alpha, alpha_error], decimals = find_decimal_place(alpha_error))

    plt.plot(results[0,:],linear_peak_width(results[0,:]*(10**9), deltaH_0, gyromagnetic_ratio), label="fit")
    plt.legend()
    plt.annotate(f"$\delta H_0 = {deltaH_0_rounded}±{deltaH_0_error_rounded}$\n$alpha = {alpha_rounded}±{alpha_error_rounded}$",
                xy=(0,results[1,-1]*0.5), xycoords='data',
                xytext=(0,results[1,-1]*0.75), textcoords='data',
                )
    plt.show()
    return deltaH_0, deltaH_0_error, alpha, alpha_error
                     

def ProcessData(filename):
    """Documentation string here.

    This function will be called with a single string parameter which will be the full name of a data file that should
    be read and processed.
    """
    # Your code goes here
    
    with open(filename, 'r') as my_data:
        for line in my_data:
            line=line.strip()
            if line == '&END':#this signals the end of the metadat 
                break
        try: #if the metadata dousnt have "&END" at the end, then raise an exception
            frequencies = next(csv.reader(my_data))
        except:
            raise Excpetion("Could not find end of metadata")
        frequencies = frequencies[0].split('\t')
        data = np.genfromtxt(my_data, unpack = True)
    differential_absorption_data = data[1:,:]#differential_absorption_data must have same numbe of columns as the number of magnetic field values
    magnetic_field = data[0,:] #because they are in an array and an array must be 'square'

    absorption_data = cumtrapz(differential_absorption_data,magnetic_field, initial = 0.0)

    results, error_widthI_0peak= results_for_all_frequ(magnetic_field, differential_absorption_data , frequencies)
    #column i in results is [frequ,width, I_0, peak], and column i in error_widthI_0peak is [width_error, I_0_error, peak_error]
    #this means that row 0 is all the frequncies, row 1 is all the peak widths, row 2 is all the I_O values and row 3 is
    #all the vaues for I_0

    index_of_20GHz = np.where(results[0,:] == 20)[0][0]
    absorption_20GHz = absorption_data[index_of_20GHz,:]
    deltaH_20GHz, I_0_20GHz, peak_20GHz = results[1:, index_of_20GHz]
    deltaH_20GHz_error, I_0_20GHz_error, peak_20GHz_error = error_widthI_0peak[:,index_of_20GHz ]
    graph_AbsorptionVMagneticField(magnetic_field, absorption_20GHz, deltaH_20GHz, I_0_20GHz, peak_20GHz, deltaH_20GHz_error, I_0_20GHz_error, peak_20GHz_error)
    
    g_factor, g_error, Hk, Hk_error, Ms, Ms_error, gamma, gamma_error= graph_FrequencyVPeakPosition(results,error_widthI_0peak[2,:])
    deltaH_0, deltaH_0_error, alpha, alpha_error = graph_PeakWidthVFrequency(results,error_widthI_0peak[0,: ], gamma, gamma_error )

    results = {
        "20GHz_peak": peak_20GHz,  # peak position at 20GHz (A m^-1)
        "20GHz_peak_error": peak_20GHz_error,  # uncertainity in above (A m^-1)
        "20GHz_width": deltaH_20GHz,  # Delta H for 20 GHz (A m^-1)
        "20GHz_width_error":deltaH_20GHz_error,  # uncertainity in above (A m^-1)
        "gamma": gamma,  # your gamma value (rad s^-1 T^-1)
        "gamma_error": gamma_error,  # uncertainity in above (rad s^-1 T^-1)
        "g": g_factor,  # Your Lande g factor (dimensionless number)
        "g_error": g_error,  # uncertainity in above (dimensionless number)
        "Hk": Hk,  # Your value for the anisotropy field (A m^-1)
        "Hk_error": Hk_error,  # uncertainity in above (A m^-1)
        "Ms": Ms,  # Your value for M_s (A m^-1)
        "Ms_error": Ms_error,  # uncertainity in above (A m^-1)
        "DeltaH": deltaH_0,  # Intrinsic line width (A m^-1)
        "DeltaH_error":deltaH_0_error,  # uncertainity in above (A m^-1)
        "alpha": alpha,  # Gilbert Damping parameter (dimensionless number)
        "alpha_error": alpha_error,
    }  # uncertainity in above (dimensionless number)

    # If your code doesn't find all of these values, just leave them set to None
    # Otherwise return the number as a floating point number (not a string or anything else).
    # Your code does not have to round these numbers corr4ectly, but you must quote them
    # rounded correctly in your report.
    return results


if __name__ == "__main__":
    # Put your test code inside this if statement to stop it being run when you import your code
    # Please avoid using input as the testing is going to be done by a computer programme, so
    # can't input things from a keyboard....
    filename = "My Data File.txt"
    test_results = ProcessData('practice_data_py20ed1.dat')
    print(test_results)

    # Before you submit, make sure that your code runs in a newly re-started console / Python kernel.
    # We strongly recommend that you use the coursework checker to make sure that your submission looks like it
    # will work. Make sure you have removed all breakpoint() statments from your code!


# In[ ]:





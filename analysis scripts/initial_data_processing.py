# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:23:32 2025

@author: jans008
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle as pkl
import subprocess
from Streak_data_analyse import analyse_streak_data
from scipy import interpolate
os.chdir(os.path.dirname(os.path.realpath(__file__)))
base_directory = os.path.dirname(os.path.realpath(__file__))
#%% defines generally usefull functions

def find_seconds(time):
    time = str(time)
    hours = time[0:2]
    minutes = time[2:4]
    seconds = time[4:6]
    time_in_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return time_in_seconds

def load_dual_PAM_measurement(file_name):
    data = pd.read_excel(file_name, skiprows=1)
    data.columns = ['Time', 'Fluo', 'P700']
    #data.file_time = data..apply(lambda x: x[9:15])
    return data


# Sp analysis equations

def calculate_Fs(measurement_frame, steady_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #Fluorescence column in V. steady_window should be list of two numbers in ms.
    #Fs is defined as the average V over the Fs time window.
    Fs_section = measurement_frame[(measurement_frame.Time > steady_window[1]-100) & (measurement_frame.Time < steady_window[1]-1)]
    Fs = Fs_section.Fluo.mean()
    return Fs


def calculate_Fm_prime(measurement_frame, SP_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #Fluorescence column in V. SP_window should be list of two numbers in ms.
    #Fm is defined as the average V over the final 100ms of the SP.
    #this approach ensures reduction of noise while removing Fm induction period
    Fm_section = measurement_frame[(measurement_frame.Time > SP_window[1]-100) & (measurement_frame.Time < SP_window[1]-1)]
    Fm_prime = Fm_section.Fluo.mean()
    return Fm_prime

def calculate_F0_prime(measurement_frame, FR_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #Fluorescence column in V. FR_window should be list of two numbers in ms.
    #F0 is defined as the average V over a period of 400 ms half a second after 
    #the start of the dark period.
    #this approach ensures reduction of noise
    F0_section = measurement_frame[(measurement_frame.Time > FR_window[1] - 1000) & (measurement_frame.Time < FR_window[1] - 500)]
    F0_prime = F0_section.Fluo.mean()
    return F0_prime

def calculate_F0(measurement_frame, steady_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #Fluorescence column in V. FR_window should be list of two numbers in ms.
    #F0 is defined as the average V over a period of 400 ms half a second after 
    #the start of the dark period.
    #this approach ensures reduction of noise
    F0_section = measurement_frame[(measurement_frame.Time > steady_window[1] - 100) & (measurement_frame.Time < steady_window[1] - 1)]
    F0_prime = F0_section.Fluo.mean()
    return F0_prime

def calculate_Ps(measurement_frame, steady_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #Ps signal column. steady_window should be list of two numbers in ms.
    #Ps is defined as the average signal over the Ps time window.

    
    Ps_section = measurement_frame[(measurement_frame.Time > steady_window[0]+20) & (measurement_frame.Time < steady_window[0]+50)]
    intercept, steepness, r_squared, p_value = calc_lin_reg(Ps_section.Time, Ps_section.P700)
    Ps = intercept + steepness * steady_window[0]
    return Ps, intercept, steepness

def calculate_P0(measurement_frame, dark_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #P700 column. dark_window should be list of two numbers in ms.
    #P0 is defined as the average signal over a period of 400 ms at the start of the dark period
    #this is shifted 100ms to easy timing with SUN-Fr system
    # This approach ensures reduction of noise
    P0_section = measurement_frame[(measurement_frame.Time > dark_window[1] - 200) & (measurement_frame.Time < dark_window[1] - 50)]
    P0 = P0_section.P700.mean()
    return P0
    
def calc_lin_reg(x,y):
    # two lists of equal length are should be provided. 
    # an ordinary least squares regression is used to get an intercept and
    #steepness value. Including two parameters to gauge goodness of fit.
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    intercept, steepness = model.params
    r_squared = model.rsquared
    p_value = model.f_pvalue
    return intercept, steepness, r_squared, p_value


def calculate_Phigh(measurement_frame, SP_window):
    #measurement frame is dataframe containing a Time column in ms and a 
    #P700 column. Start of the SP should be provided in ms
    #Phigh is defined as the hight of the P700 signal a the exact moment of the SP pulse.
    #To reduce noise a linear fit is done through the first 100ms of the P700 signal and the 
    #P700 value at start_SP of the fit is taken as the Phigh value. 
    # the first 10 ms is removed from analysis to reduce induction artifacts. 
    Phigh_section = measurement_frame[(measurement_frame.Time > SP_window[0]+10) & (measurement_frame.Time < SP_window[0]+30)]
    intercept, steepness, r_squared, p_value = calc_lin_reg(Phigh_section.Time, Phigh_section.P700)
    Phigh = intercept + steepness * SP_window[0]
    return Phigh, intercept, steepness



#Chlorophyll content calculations
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def calc_chl_concentration(wavelength_array, absorbance_array):
    # based on 
    # The Spectral Determination of Chlorophylls a and b, as well as Total Carotenoids, Using Various Solvents with Spectrophotometers of Different Resolution
    # A.R. Wellburn 1994
    # equation is for DMSO solvent with a narrow waveband spectrometer
    abs_665_1 = absorbance_array[find_nearest(wavelength_array, 665.1)]
    abs_649_1 = absorbance_array[find_nearest(wavelength_array, 649.1)]

    # calculates chl content in ug ml-1
    chl_a_ug = 12.47 * abs_665_1 - 3.62 * abs_649_1
    chl_b_ug = 25.06 * abs_649_1 - 6.5 * abs_665_1
    
    # converts to umol ml-1
    chl_a = chl_a_ug / 893.509
    chl_b = chl_b_ug / 907.492
    chl_tot = chl_a + chl_b
    
    # returns chl content in umol mL-1
    return np.array([chl_a, chl_b, chl_tot])

def calc_circle_area(diameter_mm):
    # diameter of leaf disc in mm
    
    #radius in m
    radius = 0.001 * diameter_mm * 0.5
    area = np.pi * radius * radius
    # returns area in m2
    return area

#Light spread in chamber functions
def load_calibration_measurement(file_name):
    path = file_name 
    data = pd.read_csv(path, header=1, sep=',', skiprows=7)
    data.columns = ['WL','mirror','chamber','', '','','','']
    data = data[1:]
    return data

def find_sum_between_bounds(wavelength_list, intensity_list, bounds):
    # calculate the photon flux integral between min_wl & max_wl
    x = wavelength_list
    y = intensity_list
    # find the indexes matching the range within min_wl and max_wl in
    start_index = next(i for i, xi in enumerate(x) if xi >= bounds[0])
    end_index = next(i for i, xi in enumerate(x) if xi > bounds[1])
    # crop the dataset to wl_range
    x_subset = x[start_index:end_index + 1]
    y_subset = y[start_index:end_index + 1]
    # Integrate photon flux over the wl_range with the trapezoidal rule
    pfd_integral = 0.0
    for i in range(1, len(x_subset)):
        pfd_integral += 0.5 * (x_subset[i] - x_subset[i - 1]) * (y_subset[i] + y_subset[i - 1])
    return pfd_integral

def find_sum_between_bounds_prestandardized(spectrum, bounds):
    return spectrum[bounds[0]-380:bounds[1]-380].sum()

#spectral correction functions
def nm_bin_spectrum_interpolation(wavelength_list, parameter_list):
    #TODO this function should take the spectra and interpolate the values to 
    # have values with bins of 1nm wide. 
    wavelength_list_corrected = wavelength_list
    parameter_list_corrected = parameter_list
    
    print('nm bin spectrum interpolation not implemented yet')
    return wavelength_list_corrected, parameter_list_corrected

def nm_bin_spectrum_integration(wavelength_list, parameter_list):
    wavelengths = wavelength_list
    intensities = parameter_list
    
    spectrum = np.zeros(371)
    for i in range(1,len(wavelengths)):
        bin_width = wavelengths[i] - wavelengths[i-1]
        half_under = ((wavelengths[i] + 0.5) //1) -0.5
        if wavelengths[i] - half_under > bin_width:
            wavelength = int(half_under + 0.5)
            if (wavelength >= 380 ) & (wavelength<=750):
                # light intensity is given per nm, and needs to be corrected for the bin width (of approx 0.6nm)
                spectrum[wavelength-380] += intensities[i] * bin_width
        else:
            #this section calculates the fraction of each bin that is atrributed to a lower and a higher wl
            upper_bin_fraction = (wavelengths[i] - half_under) 
            lower_bin_fraction = bin_width - upper_bin_fraction
            wavelength_lower_bin = int(half_under-0.5)
            wavelength_upper_bin = int(half_under+0.5)
            if (wavelength_lower_bin >= 380 ) & (wavelength_lower_bin<=750):
                spectrum[wavelength_lower_bin-380] += intensities[i] * lower_bin_fraction
            if (wavelength_upper_bin >= 380 ) & (wavelength_upper_bin<=750):
                spectrum[wavelength_upper_bin-380] += intensities[i] * upper_bin_fraction
    
    return spectrum
    
def standardize_spectrum(wavelength_list, parameter_list, method):
    if wavelength_list[1] - wavelength_list[0]!= 1:
        if method == 'interpolation':
            wavelength_list, parameter_list = nm_bin_spectrum_interpolation(wavelength_list, parameter_list)
        elif method == 'integration':
            parameter_list_corrected = nm_bin_spectrum_integration(wavelength_list, parameter_list)
            return parameter_list_corrected
    #this function returns the spectrum just between 380-750.
    #as these are the standard bounds that will be used for everything.
    parameter_list_corrected = np.zeros(371)
    start_wavelength = wavelength_list[0]
    final_wavelength = max(wavelength_list)

    if start_wavelength < 380:
        offset = int(380- start_wavelength)
        max_index = int(370+offset - max(0, 751-final_wavelength))
        parameter_list_corrected[0:max_index-offset] = parameter_list[offset:max_index]
    else:
        offset = int(start_wavelength-380)
        max_index = int(min(len(wavelength_list), 370-offset))
        parameter_list_corrected[offset:offset+max_index] = parameter_list[0:max_index]
        
    return parameter_list_corrected

def find_or_create_results_folder(plantID):
    folder_name = 'analysis_results'
    folder_location = os.path.join(base_directory, folder_name, plantID)
    if os.path.isdir(folder_location):
        return folder_location
    else:
        os.mkdir(folder_location)
        return folder_location
    
    
#functions for CB6F relaxation kinetics
def load_dual_PAM_measurement_CB6F(file_name):
    data = pd.read_excel(file_name, skiprows=1)
    data.columns = ['Time', 'P700']
    #data.file_time = data..apply(lambda x: x[9:15])
    return data

def calculate_dark_relax(measurement_frame, dark_window):

    dark_section = measurement_frame[(measurement_frame.Time > dark_window[0]+0) & (measurement_frame.Time < dark_window[1])]
    intercept, steepness, r_squared, p_value = calc_lin_reg(dark_section.Time, dark_section.P700)
    Phigh = intercept + steepness * dark_window[0]
    return Phigh, intercept, steepness


#functions for calculating k_N2
def calc_interpolation(filter_data, required_wavelengths, plot=False):
    interpolation_function = interpolate.interp1d(filter_data['Wavelength'], filter_data['Transmission'])
    transmission_interpolated = interpolation_function(required_wavelengths)
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(filter_data['Wavelength'], filter_data['Transmission'], color='blue', ls='-', alpha=0.5)
        ax.plot(required_wavelengths, transmission_interpolated, color='orange', ls='-', alpha=0.5)
        #fig.show()
    
    # returns transmission as a fraction per wavelenght for the required wavelengths
    return transmission_interpolated/100

def calc_params_for_NPQ(F0, Fm, streak_data, filter_data):
    # F0 and Fm will be floating values
    # streak data is a dataframe with columns (photosystem, light condition, lifetime, wavelengths and spectra)
    # order must be:
    # 1 PSII F0
    # 2 PSI F0
    # 3 PSII Fm
    # 4 PSI Fm
    # lifetime is in ps, wavelengths in nm and spectra in a.u.
    
    # filter data is a dataframe with columsn (Wavelength, Transmission)
    # wavelength is in nm and transmission in %
    
    transmission_interpolated = calc_interpolation(filter_data, streak_data.at[2, 'Wavelengths'], plot=True)
    
    Constant_PSI_0 = (streak_data.at[1,'Spectra'] * transmission_interpolated).sum() * streak_data.at[1, 'Lifetime']
    Constant_PSII_0 = (streak_data.at[0,'Spectra'] * transmission_interpolated).sum() * streak_data.at[0, 'Lifetime']
    Constant_PSI_m = (streak_data.at[3,'Spectra'] * transmission_interpolated).sum() * streak_data.at[3, 'Lifetime']
    Constant_PSII_m = (streak_data.at[2,'Spectra'] * transmission_interpolated).sum() * streak_data.at[2, 'Lifetime']
    
    a = Fm * Constant_PSII_0 - F0 * Constant_PSII_m
    b = F0 *  Constant_PSI_m + Fm * Constant_PSII_0 - F0 * Constant_PSII_m - Fm * Constant_PSI_0
    relative_PSI_excitation = a / b
    
    gain_PAM_0 = F0 / (Constant_PSI_0 * relative_PSI_excitation + Constant_PSII_0 * (1-relative_PSI_excitation))
    gain_PAM_m = Fm / (Constant_PSI_m * relative_PSI_excitation + Constant_PSII_m * (1-relative_PSI_excitation))
    
    
    PSI_contribution_Fo = Constant_PSI_0 * gain_PAM_0 * relative_PSI_excitation / F0
    PSI_contribution_Fm = Constant_PSI_0 * gain_PAM_0 * relative_PSI_excitation / Fm
    return gain_PAM_0, relative_PSI_excitation, PSI_contribution_Fo, PSI_contribution_Fm

def calc_rate_of_NPQ(Fm_prime, streak_data, filter_data, gain_PAM, relative_PSI_excitation):
    # Fm_prime is a floating value
    # streak data is a dataframe with columns (photosystem, light condition, lifetime, wavelengths and spectra)
    # order must be:
    # 1 PSII F0
    # 2 PSI F0
    # 3 PSII Fm
    # 4 PSI Fm
    # lifetime is in ps, wavelengths in nm and spectra in a.u.
    
    # filter data is a dataframe with columsn (Wavelength, Transmission)
    # wavelength is in nm and transmission in %
    
    transmission_interpolated = calc_interpolation(filter_data, streak_data.at[2, 'Wavelengths'], plot=False)
    
    
    a = (Fm_prime / gain_PAM) - (streak_data.at[3,'Spectra'] * transmission_interpolated).sum() * streak_data.at[3, 'Lifetime'] * relative_PSI_excitation
    b =  (streak_data.at[2,'Spectra'] * transmission_interpolated).sum() * (1 - relative_PSI_excitation)
    tau_PSII_m_prime = a/b

    k_NPQ = (1 / tau_PSII_m_prime) - (1 / streak_data.at[2, 'Lifetime'])
    
    #returns k_NPQ per ns
    return k_NPQ * 1000
        
#%% defines the instance for a single analysis
    
class plant_instance:
    
    def __init__(self, plantID, date, licor_start_time, lamp_start_time):
        
        self.plantID = plantID
        self.year = int(date[0:4])
        self.month = int(date[4:6])
        self.day = int(date[6:8])
        self.date_string = date
        
        self.licor_start_time = licor_start_time
        self.lamp_start_time = lamp_start_time
        
        #TODO check if these values make sense and need changing per species.
        self.gamma_star = 37.4 #ubar jans 2024, not original source
        self.K_O = 165.8 #mbar Jans 2024, not original source
        self.K_C = 272.4 #ubar jans 2024, not original source
        self.oxygen_air = 212.7
        self.g_m = 0.3 # umol m-2 s-1 ubar-1 jans 2024, not original source
        
    
    """the following functions load in all the data"""
    
    def obtain_gas_data(self, update=False):
        if update or (not hasattr(self, 'gas_exchange_data')):
            #loading gas exchange data and selecting the relevant columns in the data
            #first it is checked if the data file exists and if it is unique.
            folder_name = 'ST_measurements_licor'
            file_folder = os.path.join(base_directory, folder_name)
            file_list = [filename for filename in os.listdir(file_folder) if (self.plantID+' ' in filename) and filename.endswith('.xlsx')]
            if len(file_list) == 0:
                print(f'gas exchange file for {self.plantID} not found in {folder_name}')
                return False
            elif len(file_list) > 1:
                print(f'Too many gas exchange files found for {self.plantID}. {file_list}')
                return False
            else:
                file_location = os.path.join(file_folder, file_list[0])
                gas_frame = pd.read_excel(file_location, skiprows=14, header=0)
                gas_frame = gas_frame.iloc[1:]
                gas_frame = gas_frame.loc[:, ['hhmmss', 'obs', 'A', 'CO2_s', 'CO2_r', 'Ci', 'gsw']]
                gas_frame.reset_index(inplace=True, drop=True)
                gas_frame.hhmmss = gas_frame.hhmmss.apply(lambda x: x.replace(':', ''))
                gas_frame['seconds'] = gas_frame.hhmmss.apply(lambda x: find_seconds(x))
                
                # the following corrects the timing in the gas_frame such that the 
                # start of the lamp schedule is t=0. It corrects for both the incorrect
                # timing in the licor and the later start of the lamp then the licor
                # measurements. 
                correction_for_licor_timing = find_seconds(self.licor_start_time) - gas_frame.at[0,'seconds']
                gas_frame['seconds'] = gas_frame['seconds'] + correction_for_licor_timing
                correction_for_lamp_timing = find_seconds(self.lamp_start_time)
                gas_frame['seconds'] = gas_frame['seconds'] - correction_for_lamp_timing
                
                gas_frame = gas_frame.apply(pd.to_numeric)
    
                gas_frame['minutes']  = (gas_frame.seconds)/60
                self.gas_exchange_data = gas_frame
                return True
        else:
            return True

        
    def obtain_lamp_script(self, update=False):
        
        #loading the lamp schedule
        #first it is checked if the data file exists and if it is unique.
        if update or (not hasattr(self, 'lamp_schedule')):
            folder_name = 'lamp_scripts'
            file_folder = os.path.join(base_directory, folder_name)
            file_list = [filename for filename in os.listdir(file_folder) if (self.date_string in filename) and filename.endswith('.txt')]
            if len(file_list) == 0:
                print(f'lamp script file for {self.plantID} and {self.date_string} not found in {folder_name}')
                return False
            elif len(file_list) > 1:
                if self.date_string == '20250708' or self.date_string == '20250709':
                    filename = self.date_string + '_lamp_script_' + self.plantID[:2] + '.txt'
                    file_location = os.path.join(file_folder, filename)
                    self.lamp_schedule = pd.read_csv(file_location, sep= '\t')
                else:
                    for file_name in file_list:
                        if self.plantID[0:2] in file_name:
                            file_location = os.path.join(file_folder, file_name)
                            self.lamp_schedule = pd.read_csv(file_location, sep= '\t')
                    print(f'Multiple lamp script files found for {self.date_string}. {file_list}')
                    return False
            else:
                file_location = os.path.join(file_folder, file_list[0])
                self.lamp_schedule = pd.read_csv(file_location, sep= '\t')
                return True
            
        else:
            return True

        
    def obtain_dual_pam_files(self, plot_figures=False, update=False):
        if update or (not hasattr(self, 'dual_pam_data')):
            #first it is checked if the data folder exists and if it is unique.
            main_folder_name = 'ST_measurements_sat_flashes'
            main_folder = os.path.join(base_directory, main_folder_name)
            folder_list = [folder for folder in os.listdir(main_folder) if (self.plantID in folder)]
            if len(folder_list) == 0:
                print(f'sat flash folder for {self.plantID} not found in {main_folder_name}')
                return False
            elif len(folder_list) > 1:
                print(f'Too many sat flash folders found for {self.plantID}. {folder_list}')
                return False
            else:
                folder_location = os.path.join(main_folder, folder_list[0])
                    
                file_names = [filename for filename in os.listdir(folder_location) if filename.startswith('F') ]
                results_frame = pd.DataFrame(columns = ['filename', 'file_time', 'measurement_type', 'seconds', 'measurement_time', 'Fluo', 'P700', 'Fs', 'F0', 'Fm_prime', 'F0_prime', 'Ps', 'Phigh', 'P0'])
                results_frame.filename = file_names
                results_frame.measurement_type = ['F0_Fm', 'Pm'] + ['SP'] * (len(file_names)-2)
                results_frame.file_time = results_frame.filename.apply(lambda x: x[9:15])
    
                # DUAL pam is weird and saves the time incorrectly. so timing is shifted, and final data file is discarded as it is empty.
                for i in range(len(results_frame.filename)-3):
                    results_frame.iat[i+2, 1] = results_frame.iat[i+3, 1]
                results_frame = results_frame[:-1]
                results_frame.reset_index(inplace=True, drop=True)
                
                for i in range(len(results_frame)):
                    measurement = load_dual_PAM_measurement(os.path.join(folder_location, results_frame.at[i,'filename']))
                    
                    results_frame.at[i, 'measurement_time'] = np.array(measurement.Time)
                    results_frame.at[i, 'Fluo'] = np.array(measurement.Fluo)
                    results_frame.at[i, 'P700'] = np.array(measurement.P700)
                    
    
                    steady_window = [500,1500]
                    steady_window_F0 = [-500,500]
                    SP_window = [1500, 2300]
                    dark_window = [2300,3300]
    
                    
                    if plot_figures:
    
    
                        fig, ax = plt.subplots(1,1, figsize=(8, 5))
                        
                        ax.fill_between([(SP_window[1]-100)/1000, (SP_window[1]-1)/1000], -100, 100, color='lightcoral', alpha=0.4)
                        ax.fill_between([(dark_window[1]-200)/1000, (dark_window[1]-50)/1000], -100, 100, color='lightcoral', alpha=0.4)  
                        ax.fill_between([(steady_window[0]+20)/1000, (steady_window[0]+50)/1000], -100, 100, color='lightcoral', alpha=0.4)  
                        ax.fill_between([(SP_window[0])/1000, (SP_window[0]+100)/1000], -100, 100, color='lightcoral', alpha=0.4)

    
                        
                        
                        ax1 = ax.twinx()
                        line = calculate_Phigh(measurement, SP_window)[1] + calculate_Phigh(measurement,SP_window)[2] * np.array(measurement.Time)
                        P700, = ax1.plot(measurement['Time']/1000, measurement['P700'], c='orange', label='PSI', alpha=0.5)
                        ax1.set_ylim(min(measurement['P700'])-0.5, max(measurement['P700'])+0.5)
                        
                        Fluo, = ax.plot(measurement['Time']/1000, measurement['Fluo'], c='b', label='PSII')
                        ax.set_xlim(min(measurement.Time)/1000, max(measurement['Time']/1000))
                        ax.set_ylim(0, 0.3)
    
    
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Fluorescence signal (V)')
                        ax1.set_ylabel('P700 deltaI/I *1000')
                        #ax.set_title('Analysis of a single Saturating Pulse time: '+str(results_frame.file_time[i]) + ' type: ' + results_frame.measurement_type[i])
                        ax.legend(handles=[Fluo, P700], loc='upper left')
                        
                        axin = ax.inset_axes([0.7, 0.6, 0.25, 0.3], transform=ax.transAxes)
    
                        axin.plot(measurement['Time']/1000, line, c='black', label = 'Phigh_fit')
                        axin.plot(measurement['Time']/1000, measurement['P700'], c='orange', label='PSI')
    
                        axin.set_xticks([SP_window[0]/1000, SP_window[0]/1000+ 0.1])
                        axin.set_xlim(SP_window[0]/1000, SP_window[0]/1000+ 0.1)
                        axin.set_yticks([])
                        axin.set_ylim(min(measurement['P700'])-1, max(measurement['P700'])+1)
    
                        axin.xaxis.set_tick_params(labelsize=7)
                        axin.set_xlabel('Time (s)', fontsize=7)
                        axin.yaxis.set_tick_params(labelsize=7)
                        axin.spines[['right', 'top']].set_visible(False)
                        axin.set_title('Pm\' zoom', fontsize=7)
                        
                        axin2 = ax.inset_axes([0.4, 0.6, 0.25, 0.3], transform=ax.transAxes)
                        line2 = calculate_Ps(measurement, steady_window)[1] + calculate_Ps(measurement,steady_window)[2] * np.array(measurement.Time)
    
                        axin2.plot(measurement['Time']/1000, line2, c='black', label = 'Ps_fit')
                        axin2.plot(measurement['Time']/1000, measurement['P700'], c='orange', label='PSI')
    
                        axin2.set_xticks([steady_window[0]/1000, steady_window[0]/1000+ 0.1])
                        axin2.set_xlim(steady_window[0]/1000, steady_window[0]/1000+ 0.1)
                        axin2.set_yticks([])
                        axin2.set_ylim(min(measurement['P700'])-1, max(measurement['P700'])+1)
    
                        axin2.xaxis.set_tick_params(labelsize=7)
                        axin2.set_xlabel('Time (s)', fontsize=7)
                        axin2.yaxis.set_tick_params(labelsize=7)
                        axin2.spines[['right', 'top']].set_visible(False)
                        axin2.set_title(r'$P700_{ox}$ zoom', fontsize=7)
                        
                        
                        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
                        ax.annotate('Fm & Fm\'', xy=(2.250, 0.02), xytext=(2.5, 0.02), transform=ax.transAxes, fontsize=7, 
                                    bbox=props, arrowprops=dict(arrowstyle = '-', facecolor='black'))
                        ax.annotate(r'$P700_{ox}$', xy=(0.53, 0.02), xytext=(1, 0.02), transform=ax.transAxes, fontsize=7, 
                                    bbox=props, arrowprops=dict(arrowstyle = '-', facecolor='black'))
                        ax.annotate('P0', xy=(3.15, 0.02), xytext=(4.0, 0.02), transform=ax.transAxes, fontsize=7, 
                                    bbox=props, arrowprops=dict(arrowstyle = '-', facecolor='black'))
                        ax.annotate('Pm', xy=(1.5, 0.02), xytext=(1.7, 0.02), transform=ax.transAxes, fontsize=7, 
                                    bbox=props, arrowprops=dict(arrowstyle = '-', facecolor='black'))
                       
                        
                        fig.tight_layout()
                        save_name = os.path.join(folder_location,f'{results_frame.at[i,"filename"][9:15]}_SP_analysis.png')
                        fig.savefig(save_name,dpi=400)
                        plt.close()
                    
                    if i == 0 :
                        results_frame.loc[i, 'F0'] = calculate_F0(measurement, steady_window=steady_window_F0)
                    else:
                        results_frame.loc[i, 'F0'] = calculate_F0(measurement, steady_window=steady_window)
    
                    results_frame.loc[i, 'Fs'] = calculate_Fs(measurement, steady_window=steady_window)
                    results_frame.loc[i, 'Fm_prime'] = calculate_Fm_prime(measurement, SP_window=SP_window)
                    results_frame.loc[i, 'Ps'] = calculate_Ps(measurement, steady_window=steady_window)[0]
                    results_frame.loc[i, 'Phigh'] = calculate_Phigh(measurement, SP_window=SP_window)[0]
                    results_frame.loc[i, 'P0'] = calculate_P0(measurement, dark_window=dark_window)
    
                self.Fm = results_frame.loc[0, 'Fm_prime'] 
                self.Pm = results_frame.loc[1, 'Phigh'] - results_frame.loc[1, 'P0']
                self.F0 = results_frame.loc[0, 'F0']
                self.Fv_Fm = (self.Fm-self.F0)/self.Fm
    
                results_frame.seconds = results_frame.file_time.apply(lambda x: find_seconds(x)) 
                
                correction_for_lamp_timing = find_seconds(self.lamp_start_time)
                results_frame['seconds'] = results_frame['seconds'] - correction_for_lamp_timing
                
                results_frame['minutes']  = (results_frame.seconds)/60
                self.dual_pam_data = results_frame
                if self.Fv_Fm < 0.7:
                    print(f'Fv/Fm of {self.plantID} is {self.Fv_Fm}. Please check if plant is healthy and analysis is correct.')
                return True

        else:
            return True




    
    def obtain_incident_spectra(self, plot_figures=False, update=False):
        if update or (not hasattr(self, 'incident_spectra')):
            main_folder_name = 'incident_light_spectra'
            main_folder = os.path.join(base_directory, main_folder_name)
            folder_list = [folder for folder in os.listdir(main_folder) if (self.date_string in folder) and not folder.endswith('.png')]
            if len(folder_list) == 0:
                print(f'light spread chamber folder for {self.plantID} and {self.date_string} not found in {main_folder_name}')
                return False
            elif len(folder_list) > 1:
                print(f'Too many light spread chamber folders found for {self.plantID} and {self.date_string}. {folder_list}')
                return False
            else:
                folder_location = os.path.join(main_folder, folder_list[0])
                file_names = [filename for filename in os.listdir(folder_location) if filename.endswith('.csv') ]
                spectra_frame = pd.DataFrame(columns = ['filename', 'wavelengths', 'PFD_chamber', 'PAR', 'FR', 'light_filter'])
    
                spectra_frame.filename = file_names
                PAR_list = []
                FR_list =[]
                filter_list = []
                for name in file_names:
                    name_splitted = name.split('_')
                    PAR_list.append(float(name_splitted[2].replace(' ', '.')))
                    FR_list.append(float(name_splitted[3]))
                    filter_list.append(int(name_splitted[4][0:1]))
                    
                spectra_frame.PAR = PAR_list
                spectra_frame.FR = FR_list
                spectra_frame.light_filter = filter_list
    
                for i in range(len(spectra_frame)):
                    measurement = load_calibration_measurement(os.path.join(folder_location,file_names[i]))
    
                    spectra_frame.at[i, 'wavelengths'] = np.array(pd.to_numeric(measurement.WL))
                    spectra_frame.at[i, 'PFD_chamber'] = np.array(pd.to_numeric(measurement.chamber))
    
                spectra_frame['chamber_PAR'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [400,700]), axis=1)
                spectra_frame['chamber_FR'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [700,750]), axis=1)
                spectra_frame['chamber_tot'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [400,750]), axis=1)
                #wavelengths = np.array(spectra_frame.at[0, 'wavelengths'])
                
                index_WL = int(spectra_frame.index[(spectra_frame.PAR ==1.9) & (spectra_frame.FR == 0)].tolist()[0])
                index_WL_FR = int(spectra_frame.index[(spectra_frame.PAR ==1.5) & (spectra_frame.FR == 10)].tolist()[0])
                
                self.incident_spectra = spectra_frame
                
                #correct incident light spectra for difference between hole measurement and light spread measurement
                index_correction =  int(spectra_frame.index[(spectra_frame.PAR ==2) & (spectra_frame.FR == 2)].tolist()[0])
                correction_spectrum = self.chamber_average_spectrum / spectra_frame.at[index_correction, 'PFD_chamber']
                
                self.WL_spectrum = standardize_spectrum(spectra_frame.at[index_WL, 'wavelengths'], correction_spectrum * spectra_frame.at[index_WL, 'PFD_chamber'], 'integration')
                self.WL_PAR = find_sum_between_bounds_prestandardized(self.WL_spectrum, [400,701])
                
                self.WL_FR_spectrum = standardize_spectrum(spectra_frame.at[index_WL_FR, 'wavelengths'], correction_spectrum * spectra_frame.at[index_WL_FR, 'PFD_chamber'], 'integration')
                self.WL_FR_PAR = find_sum_between_bounds_prestandardized(self.WL_FR_spectrum, [400,701])

                if plot_figures:
                    fig, ax = plt.subplots(1,1, figsize =  (5,5))
                    WL, = ax.plot(np.arange(380,751), self.WL_spectrum, ls='-', marker='', lw=2,c='blue', label='WL')
                    WLFR, = ax.plot(np.arange(380,751), self.WL_FR_spectrum, ls='-', marker='',lw=2, c='orange', label='WL+FR')
                    ax.set_xlabel(r'Wavelengths (nm)')
                    ax.set_ylabel(r'LIght intensity ($\mu mol\ m^{-2}\ s^{-1}$)')
                    ax.set_xlim(380,750)
                    ax.set_ylim(0,max(self.WL_FR_spectrum))
                    ax.set_title(f'Incident light spectra {self.plantID}')
                    ax.legend(handles=[WL, WLFR], loc='upper left')
                    fig.tight_layout()
                    save_name = os.path.join(main_folder,f'{self.plantID}_Incident light.png')
                    fig.savefig(save_name,dpi=400)
                    plt.close()
    
            return True
        else:
            return True
    

    def load_and_calculate_chl_content(self, DMSO_volume=5, DMSO_dilution_after_extraction=1,
                                       diameter_leaf_disc_mm = 10, num_discs = 2, update=False):
        if update or (not hasattr(self, 'chl_ab_ratio')):
            #load the absorbance file of the chl in DMSO extraction.
            #Standard two discs of 10mm diameter are used in 5 ml DMSO. 
            #first it is checked if the data file exists and if it is unique.
            folder_name = 'chlorophyll_content_measurements'
            file_folder = os.path.join(base_directory, folder_name)
            file_list = [filename for filename in os.listdir(file_folder) if (self.plantID in filename) and filename.endswith('.txt')]
            if len(file_list) == 0:
                print(f'chl content file for {self.plantID} not found in {folder_name}')
                return False
            elif len(file_list) > 1:
                print(f'Too many chl content files found for {self.plantID}. {file_list}')
                return False
            else:
                file_location = os.path.join(file_folder, file_list[0])
                
                with open(file_location, 'r') as file:
                    # Skip the first 14 rows (header) in the ASCII file
                    lines = file.readlines()[14:]
                    data = np.array([list(map(lambda x: float(x.replace(',', '.')), line.split())) for line in lines])
                    
                    chl_wavelength = data.T[0]
                    chl_absorbance = data.T[1]
                
                chl_concentration_DMSO = calc_chl_concentration(chl_wavelength, chl_absorbance)
                leaf_area = calc_circle_area(diameter_leaf_disc_mm) * num_discs  # in m2
                
                chl_content_leaf = chl_concentration_DMSO * DMSO_volume * DMSO_dilution_after_extraction / leaf_area
                self.chl_a_leaf = chl_content_leaf[0] 
                self.chl_b_leaf = chl_content_leaf[1] 
                self.chl_tot_leaf = chl_content_leaf[2]
                self.chl_ab_ratio = chl_content_leaf[0]/chl_content_leaf[1]
                
                if (self.chl_ab_ratio <2.5) or (self.chl_ab_ratio >3.5):
                    print(f'chlorophyll a/b ratio for {self.plantID} is {self.chl_ab_ratio}. Please check if it is correct.')
        else:
            return True
                    
    
    def obtain_Aci_data(self, plot_figures=False, update=False):
        if update or (not hasattr(self, 'Aci_data')):
            #loading gas exchange data and selecting the relevant columns in the data
            #first it is checked if the data file exists and if it is unique.
            folder_name = 'Aci_curves'
            file_folder = os.path.join(base_directory, folder_name)
            file_list = [filename for filename in os.listdir(file_folder) if (self.plantID in filename) and filename.endswith('.xlsx')]
            if len(file_list) == 0:
                print(f'Aci file for {self.plantID} not found in {folder_name}')
                return False
            elif len(file_list) > 1:
                print(f'Too many Aci files found for {self.plantID}. {file_list}')
                return False
            else:
                file_location = os.path.join(file_folder, file_list[0])
                gas_frame = pd.read_excel(file_location, skiprows=14, header=0)
                gas_frame = gas_frame.iloc[1:]
                gas_frame = gas_frame.loc[:, ['hhmmss', 'obs', 'A', 'CO2_s', 'CO2_r', 'Ci']]
                gas_frame.reset_index(inplace=True, drop=True)
                gas_frame.hhmmss = gas_frame.hhmmss.apply(lambda x: x.replace(':', ''))
                gas_frame['seconds'] = gas_frame.hhmmss.apply(lambda x: find_seconds(x))
                          
                gas_frame = gas_frame.apply(pd.to_numeric)
    
                gas_frame['minutes']  = (gas_frame.seconds)/60
                
                # removes final measurement as TPU limitation is induced and assimilation is low
                gas_frame = gas_frame.iloc[:-1]
                #also removes extra measurements at 450 ppm.
                gas_frame = gas_frame.drop([5,7])
                #also removes datapoints if the final measurement is lower then the previous one. 
                if gas_frame.iat[-1,2] < gas_frame.A.max():
                    gas_frame = gas_frame.iloc[:-1]
                if gas_frame.iat[-1,2] < gas_frame.A.max():
                    gas_frame = gas_frame.iloc[:-1]
                
                
                #analysis of the Aci curves is done according to the linearisation 
                # procedure of Long and Bernacchi 2003 for Vcmax

                gas_frame['Ci_ubar'] = gas_frame.Ci * 1.01
                gas_frame['Cc'] = gas_frame.Ci_ubar - (gas_frame.A/self.g_m)
                a = gas_frame.Cc - self.gamma_star
                b = gas_frame.Cc + self.K_C * (1+ self.oxygen_air/self.K_O)
                gas_frame['f_prime'] = a/b

                
                Rday_fit_vcmax, Vcmax_fit, rest_var, rest_var2 = calc_lin_reg(gas_frame.loc[0:4,].f_prime, gas_frame.loc[0:4,].A)
                
                self.Vcmax_lin = Vcmax_fit
                self.Rday_lin = Rday_fit_vcmax
                if plot_figures:
                    fig, ax = plt.subplots(1,2, figsize =  (10,5))
                    ax[0].plot(gas_frame.Cc, gas_frame.A, ls='', marker='o', c='orange')
                    ax[0].set_xlabel(r'Cc ($\mu mol mol^{-1}$)')
                    ax[0].set_ylabel(r'Assimilation ($\mu mol\ m^{-2}\ s^{-1}$)')
                    ax[0].set_xlim(0, max(gas_frame.Ci)+20)
                    ax[0].set_ylim(-2,max(gas_frame.A)+2)
                    ax[0].set_title(f'Aci-curve for {self.plantID}')
                    
                    ax[1].plot(gas_frame.f_prime, gas_frame.A, ls='', marker='o', c='orange')
                    f_prime_range = np.linspace(min(gas_frame.f_prime), max(gas_frame.f_prime), 100)
                    fitted = Rday_fit_vcmax + Vcmax_fit * f_prime_range
                    ax[1].plot(f_prime_range, fitted, ls='-', marker='', c='blue')
                    ax[1].set_xlabel(r'f\'')
                    ax[1].set_ylabel(r'Assimilation ($\mu mol\ m^{-2}\ s^{-1}$)')
                    ax[1].set_xlim(min(gas_frame.f_prime), max(gas_frame.f_prime))
                    ax[1].set_ylim(-2,max(gas_frame.A)+2)
                    ax[1].set_title(f'Vcmax fit for {self.plantID}')
                    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
                    string = r'$R_{day}$ = ' + str(round(Rday_fit_vcmax, 3)) + ' \n' + r'$V_{cmax}$ = ' + str(round(Vcmax_fit, 3))
                    ax[1].text(0.60, 0.1, string, transform=ax[1].transAxes, fontsize=11, bbox=props)

                    
                    fig.tight_layout()
                    save_name = os.path.join(file_folder,f'{self.plantID}_Aci_curve.png')
                    fig.savefig(save_name,dpi=400)
                    plt.close()
                
                #In addition Aci analysis is done in R with the fit aci function from plant ecophys
                #TODO should we provide the R_day value as input for the determination of Vcmax and Jma
                # both for the fitaci analysis as the linearisation analysis?
                command = r"C:/Program Files/R/R-4.4.1/bin/Rscript.exe"
                path2script = r"Aci_curves/fitaci_analysis.R"
                
                args = list(gas_frame.A) + list(gas_frame.Ci)
                for i in range(len(args)):
                    args[i] = str(args[i])
                cmd = [command, path2script] + args + [self.plantID]


                result = subprocess.check_output(cmd, universal_newlines=True)
                Vcmax, Jmax, Rday = result[26:].split(' ')
                self.Vcmax_planteco = Vcmax
                self.Jmax_planteco = Jmax
                self.Rday_planteco = Rday
                #print(f'{self.plantID}: Vcmax{Vcmax} Jmax{Jmax} Rday{Rday}')
                self.Aci_data = gas_frame
                return True
                
        else:
            return True
        
    def load_and_check_light_spread_chamber(self, plot_figures=False, update=False):
        #first it is checked if the data file exists and if it is unique.
        if update or (not hasattr(self, 'light_spread_analysed')):
            self.light_spread_analysed=True
            
            main_folder_name = 'light_spread_chamber'
            main_folder = os.path.join(base_directory, main_folder_name)
            folder_list = [folder for folder in os.listdir(main_folder) if (self.date_string in folder) and not folder.endswith('.png')and not folder.endswith('.csv')]
            if len(folder_list) == 0:
                print(f'light spread chamber folder for {self.plantID} and {self.date_string} not found in {main_folder_name}')
                return False
            elif len(folder_list) > 1:
                print(f'Too many light spread chamber folders found for {self.plantID} and {self.date_string}. {folder_list}')
                return False
            else:
                folder_location = os.path.join(main_folder, folder_list[0])
                    
                file_names = [filename for filename in os.listdir(folder_location) if filename.endswith('.csv') ]
                if len(file_names) != 25:
                    print(f'the number of light spread files in {folder_location} is {len(file_names)}. This should be 25 so check.')
                
                spectra_frame = pd.DataFrame(columns = ['filename', 'wavelengths', 'PFD_chamber', 'PAR', 'FR', 'filter', 'location'])
                spectra_frame.filename = file_names
                spectra_frame.PAR = [2]*len(file_names)
                spectra_frame.FR = [2]*len(file_names)
                spectra_frame.filter = [1]*len(file_names)
                spectra_frame.location = list(range(1,len(file_names)+1)) 

                for i in range(len(file_names)):
                    measurement = load_calibration_measurement(os.path.join(folder_location,file_names[i]))

                    spectra_frame.at[i, 'wavelengths'] = np.array(pd.to_numeric(measurement.WL))
                    spectra_frame.at[i, 'PFD_chamber'] = np.array(pd.to_numeric(measurement.chamber))

                spectra_frame['chamber_PAR'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [400,700]), axis=1)
                spectra_frame['chamber_FR'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [700,750]), axis=1)
                spectra_frame['chamber_tot'] = spectra_frame.apply(lambda x: find_sum_between_bounds(x.wavelengths, x.PFD_chamber, [400,750]), axis=1)
                wavelengths = np.array(spectra_frame.at[0, 'wavelengths'])
                average_spectrum = np.mean(np.array(spectra_frame.PFD_chamber), axis=0)
                self.chamber_average_spectrum = average_spectrum
                if (plot_figures == True) & (len(file_names)==25):
                    fig, ax = plt.subplots(1,3, figsize=(12, 4))

                    for j in range(len(spectra_frame.PAR)):
                        ax[0].plot(wavelengths, spectra_frame.at[j,'PFD_chamber'], c='blue', alpha=0.5, lw=1)#c=colors[j])

                    ax[0].set_xlim(min(wavelengths), 750)
                    ax[0].set_xlabel('Wavelength (nm)')
                    ax[0].set_ylabel(r'Light intensity ($\mu$mol $m^{-2}\ s^{-1}\ nm^{-1}$)')
                    ax[0].set_title('within chamber')
              
                    stacked = np.vstack(spectra_frame['PFD_chamber'])
                    std_list = np.std(stacked, axis=0)
                    mean_list = np.mean(stacked, axis=0)

                    ax[1].plot(wavelengths, (mean_list*mean_list)/(std_list*std_list), c='lightcoral')
                    ax[1].set_title('SNR' )

                    ax[1].set_xlim(min(wavelengths), 750)
                    ax[1].set_ylim(0,1000)

                    ax[1].set_xlabel('Wavelength (nm)')
                    ax[1].set_ylabel(r'mean$^2$/std$^2$ Light intensity (-)')

                    im0 = ax[2].imshow(np.array(spectra_frame.loc[:24, 'chamber_PAR']).reshape((5,5)))

                    cbar = fig.colorbar(im0, orientation='vertical')  # ax[:] spans both axes
                    cbar.ax.set_ylabel('PAR')

                    ax[2].set_title('PAR at different locations')

                    fig.tight_layout()
                    save_name = os.path.join(main_folder,f'{self.date_string}_light_spread_in_chamber.png')
                    fig.savefig(save_name,dpi=400)
                    plt.close()
                    
                    save_name = os.path.join(main_folder,f'{self.date_string}_light_spread_array.csv')
                    np.savetxt(save_name, np.array(spectra_frame.loc[:24, 'chamber_PAR']), delimiter=",")
                    


            
    def load_and_analyse_CB6F_rates(self, update=False):

        #TODO
        if update or (not hasattr(self, 'CB6F_frame')):
            main_folder_name = 'CB6F_rate_measurements'
            main_folder = os.path.join(base_directory, main_folder_name)
            folder_list = [folder for folder in os.listdir(main_folder) if (self.plantID in folder) and not folder.endswith('.png')]
            if len(folder_list) == 0:
                print(f'CB6F folder for {self.plantID} and {self.date_string} not found in {main_folder_name}')
                return False
            elif len(folder_list) > 1:
                print(f'Too many CB6F folders found for {self.plantID} and {self.date_string}. {folder_list}')
                return False
            else:
                folder_location = os.path.join(main_folder, folder_list[0])
                measurement_names = [filename for filename in os.listdir(folder_location) if filename.endswith('.xlsx') ]

                #final measurement did not save the data so it is discarded. 
                measurement_names = measurement_names[:-1]
    
                results_frame = pd.DataFrame(columns = ['filename', 'file_time', 'measurement_type', 'seconds', 'measurement_time', 'P700', 'Ps', 'Phigh', 'P0'])
                results_frame.filename = measurement_names
                results_frame.measurement_type = ['Pm'] + ['rereduction'] * (len(measurement_names)-1)
                results_frame.file_time = results_frame.filename.apply(lambda x: x[9:15])
                results_frame.reset_index(inplace=True, drop=True)
                
                for i in range(len(results_frame)):
                    measurement = load_dual_PAM_measurement_CB6F(os.path.join(folder_location, results_frame.at[i,'filename']))
                    
                    results_frame.at[i, 'measurement_time'] = np.array(measurement.Time)
                    results_frame.at[i, 'P700'] = np.array(measurement.P700)
                    
                    if i == 0:
                        steady_window = [-500,1500]
                        SP_window = [1500, 2300]
                        dark_window = [2300,3500]
                        
                        #plots Po and Phigh measurement
                        plot = True
                        if plot:
                    
                    
                            fig, ax = plt.subplots(1,1, figsize=(8, 5))
                            
                            ax.fill_between([(dark_window[1]-200)/1000, (dark_window[1]-50)/1000], -100, 100, color='lightcoral', alpha=0.4)  
                            
                    
                            
                            line = calculate_Phigh(measurement, SP_window)[1] + calculate_Phigh(measurement,SP_window)[2] * np.array(measurement.Time)
                            P700_fit, = ax.plot(measurement['Time']/1000, line, c='grey', label = 'Phigh_fit')
                            P700, = ax.plot(measurement['Time']/1000, measurement['P700'], c='orange', label='PSI')
                            ax.set_ylim(min(measurement['P700'])-0.5, max(measurement['P700'])+0.5)
                            
                    
                    
                    
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('P700 deltaI/I *1000')
                            #ax.set_title('Analysis of a single Saturating Pulse time: '+str(results_frame.file_time[i]) + ' type: ' + results_frame.measurement_type[i])
                            ax.legend(handles=[ P700, P700_fit], loc='upper left')
                            
                            axin = ax.inset_axes([0.7, 0.6, 0.25, 0.3], transform=ax.transAxes)
                    
                            axin.plot(measurement['Time']/1000, line, c='black', label = 'Phigh_fit')
                            axin.plot(measurement['Time']/1000, measurement['P700'], c='orange', label='PSI')
                    
                            axin.set_xticks([SP_window[0]/1000, SP_window[0]/1000+ 0.1])
                            axin.set_xlim(SP_window[0]/1000, SP_window[0]/1000+ 0.1)
                            axin.set_yticks([])
                            axin.set_ylim(min(measurement['P700'])-1, max(measurement['P700'])+1)
                    
                            axin.xaxis.set_tick_params(labelsize=7)
                            axin.set_xlabel('Time (s)', fontsize=7)
                            axin.yaxis.set_tick_params(labelsize=7)
                            axin.spines[['right', 'top']].set_visible(False)
                            axin.set_title('P700 zoom', fontsize=7)
                            
                            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    
    
                            ax.annotate('P0', xy=(3.4, 0.02), xytext=(4.0, 0.02), transform=ax.transAxes, fontsize=7, 
                                        bbox=props, arrowprops=dict(arrowstyle = '-', facecolor='black'))
    
                    
                           
                            
                            fig.tight_layout()
                            save_name = os.path.join(folder_location,f'{results_frame.at[i,"filename"]}.png')
                            fig.savefig(save_name,dpi=400)
                            plt.close()
                    else:
                        fig, ax = plt.subplots(1,1, figsize=(8, 5))
                        
                        ax.fill_between([0,5], -100, 100, color='lightcoral', alpha=0.4)  
                        #this data looks weird, and possible improves the fit with lower fitting range.
                        if self.plantID == 'SF4':
                            dark_window = [0,0.3]
                        else:
                            dark_window = [0,2]
                        
                        line = calculate_dark_relax(measurement, dark_window)[1] + calculate_dark_relax(measurement,dark_window)[2] * np.array(measurement.Time)
                        P700_fit, = ax.plot(measurement['Time'], line, c='grey', label = 'relaxation_fit')
                        P700, = ax.plot(measurement['Time'], measurement['P700'], c='orange', label='PSI')
                        ax.set_ylim(min(measurement['P700'])-0.5, max(measurement['P700'])+0.5)
                        
    
                        ax.set_xlabel('Time (ms)')
                        ax.set_ylabel('P700 deltaI/I *1000')
                        ax.set_title('Analysis of a single Saturating Pulse time: '+str(results_frame.file_time[i]) + ' type: ' + results_frame.measurement_type[i])
                        ax.legend(handles=[ P700, P700_fit], loc='upper left')
                        
                        fig.tight_layout()
                        save_name = os.path.join(folder_location,f'{results_frame.at[i,"filename"]}.png')
                        fig.savefig(save_name,dpi=400)
                        plt.close()
    
                    if i == 0 :
                            
                        results_frame.loc[i, 'Phigh'] = calculate_Phigh(measurement, SP_window=SP_window)[0]
                        results_frame.loc[i, 'P0'] = calculate_P0(measurement, dark_window=dark_window)
                    else:
    
                        results_frame.loc[i, 'reduction_slope'] = calculate_dark_relax(measurement,dark_window)[2]
                        
                #%%
                results_frame['SP int'] = [0, 3,5,7,9,11,13,15,17]
                results_frame['PAR'] = [0, 28, 78, 149, 272, 455, 722, 1110, 1723]
                Pm = results_frame.at[0, 'Phigh'] - results_frame.at[0,'P0']
                results_frame['Pm'] = Pm
                results_frame = results_frame[1:]
                results_frame.reset_index(inplace=True)
                #walz manual p 56.
                results_frame['PSI rate']  = results_frame['reduction_slope'] * 1000 * (1/Pm) * -1
                self.CB6F_frame = results_frame

                return True
        else:
            return True
    
    def load_leaf_absorbance(self, plot_figures=False, update=False):
        if update or (not hasattr(self, 'leaf_absorbance')):
            #load the leaf absorbance.
            #first it is checked if the data file exists and if it is unique.
            folder_name = 'leaf_absorbance_measurements'
            file_folder = os.path.join(base_directory, folder_name)
            file_list = [filename for filename in os.listdir(file_folder) if (self.plantID in filename) and (not filename.endswith('.png'))]
            if len(file_list) == 0:
                print(f'leaf abs file for {self.plantID} not found in {folder_name}')
                return False
            elif len(file_list) > 1:
                print(f'Too many leaf abs files found for {self.plantID}. {file_list}')
                return False
            else:
                file_location = os.path.join(file_folder, file_list[0])
                
                with open(file_location, 'r') as file:
                    # Skip the first 14 rows (header) in the ASCII file
                    lines = file.readlines()[1:]
                    data = np.array([list(map(lambda x: float(x.replace(',', '.')), line.split('\t'))) for line in lines])
                    leaf_abs_wavelength = data.T[8]
                    leaf_absorbance = data.T[11]
    
                    self.leaf_absorbance = standardize_spectrum(leaf_abs_wavelength, leaf_absorbance, 'interpolation')
                    
                if plot_figures:
                    fig, ax = plt.subplots(1,1, figsize =  (5,5))
                    absorbance, = ax.plot(np.arange(380,751), self.leaf_absorbance, ls='-', marker='', lw=2,c='blue', label='leaf abs')
                    ax.set_xlabel(r'Wavelengths (nm)')
                    ax.set_ylabel('leaf level absorptance (-)')
                    ax.set_xlim(380,750)
                    ax.set_ylim(0,1.05)
                    ax.set_title(f'leaf absorptance spectrum {self.plantID}')
                    ax.legend(handles=[absorbance], loc='upper left')
                    fig.tight_layout()
                    save_name = os.path.join(file_folder,f'{self.plantID}_leaf absorptance.png')
                    fig.savefig(save_name,dpi=400)
                    plt.close()
                    
                return True
        else:
            return True

    def load_streak_measurements(self, update=False):
        if not update and (hasattr(self, 'k_P2')):
            return True
        main_folder_name = 'Streak_measurements/Results'
        main_folder = os.path.join(base_directory, main_folder_name)
        folder_list = [folder for folder in os.listdir(main_folder) if (self.plantID in folder) and not folder.endswith('.img')]
        if len(folder_list) == 0:
            print(f'Streak folder for {self.plantID} and {self.date_string} not found in {main_folder_name}')
            return False
        elif len(folder_list) > 1:
            print(f'Too many streak folders found for {self.plantID} and {self.date_string}. {folder_list}')
            return False
        else:
            folder_location = os.path.join(main_folder, folder_list[0])
            k_P2, k_D2, PSI_PSII_ratio_streak, streak_data_for_NPQ = analyse_streak_data(folder_location, self.plantID)
            
            self.k_P2 = k_P2
            self.k_D2 = k_D2
            self.PSI_PSII_ratio_streak = PSI_PSII_ratio_streak
            print('STREAK:'+ self.plantID + 'excitation ratio'+ str(PSI_PSII_ratio_streak))
            #TODO nadenken of ik absorbtie correctie er weer uit wil halen voor de NPQ calculatie
            self.streak_data_for_NPQ = streak_data_for_NPQ
        return True
   

    
    def analyse_k_NPQ(self, update=False):
        if not update and (hasattr(self, 'k_N2')):
            return True
        
        streak_data = self.streak_data_for_NPQ

        # data taken from: https://www.pgo-online.info/4dcgi/schott-filter-calculator/
        # RG9 filter with 3mm thickness
        filter_data = pd.read_csv('RG9_spectrum.csv', sep=',', decimal='.', names=['Wavelength', 'Transmission'])

        F0 = self.F0
        Fm = self.Fm
        Fm_prime = self.dual_pam_data.at[8,'Fm_prime']

        # this calculation takes into account the effect of fluorescence from PSI.
        # It assumes the excitation balance is the same under F0, Fm and Fm' conditions. 
        #  it also assumes the gain per photon in the PAM is the same under each condition.
        gain_PAM, relative_PSI_excitation, PSI_contribution_Fo, PSI_contribution_Fm = calc_params_for_NPQ(F0, Fm, streak_data, filter_data)
        k_NPQ = calc_rate_of_NPQ(Fm_prime, streak_data, filter_data, gain_PAM, relative_PSI_excitation)

        #print(f'k_NPQ of {self.plantID} is: {k_NPQ:.2f}')
        # calculating NPQ without PSi correction. 
        NPQ = (Fm - Fm_prime) / Fm_prime
        k_NPQ_uncorrected = NPQ * streak_data.at[2, 'Lifetime']/1000 # k_NPQ in ns-1

        #calculating epsilon for PAM analysis
        #epsilon of the model is calculated by multiplying the area normalised fluorescence spectrum 
        # with the transmission spectrum of the RG9 filter
        transmission_interpolated = calc_interpolation(filter_data, streak_data.at[2, 'Wavelengths'])
        PSII_fluorescence_norm = streak_data.at[2,'Spectra'] / streak_data.at[2,'Spectra'].sum()
        PSI_fluorescence_norm = streak_data.at[3,'Spectra'] / streak_data.at[3,'Spectra'].sum()
        epsilon_PSII = (PSII_fluorescence_norm * transmission_interpolated).sum()
        epsilon_PSI = (PSI_fluorescence_norm * transmission_interpolated).sum()
        #print(f'epsilon_PSII/PSI of {self.plantID} is: {epsilon_PSII:.2f}/{epsilon_PSI:.2f}')
        
        #checks if reabsorbtion corrections matters for calculating epsilon
        #conclusion: it does not. so everything is calculated with reabsorbtion correction. 
        #PSII_fluorescence_norm_uncor = streak_data.at[2,'Spectra_uncorrected'] / streak_data.at[2,'Spectra_uncorrected'].sum()
        #PSI_fluorescence_norm_uncor = streak_data.at[3,'Spectra_uncorrected'] / streak_data.at[3,'Spectra_uncorrected'].sum()
        #epsilon_PSII_uncor = (PSII_fluorescence_norm_uncor * transmission_interpolated).sum()
        #epsilon_PSI_uncor = (PSI_fluorescence_norm_uncor * transmission_interpolated).sum()
        #print(f'epsilon_PSII/PSI of {self.plantID} is: {epsilon_PSII_uncor:.2f}/{epsilon_PSI_uncor:.2f}')
                
        self.k_N2 = k_NPQ
        self.k_N2_no_PSI_cor = k_NPQ_uncorrected
        self.epsilon_PSII = epsilon_PSII
        self.epsilon_PSI = epsilon_PSI
        return True

    """the following functions start analysing the data"""
    def remove_flashes_from_gas_exchange_data(self, update=False):
        if update or (not hasattr(self, 'gas_exchange_data_no_SP')):
            
            if hasattr(self, 'gas_exchange_data') & hasattr(self, 'lamp_schedule'):
                #select gas exchange data, but remove 1 minute after every SP
                #finds timings of the saturating flashes
                SP_timings = []
                for i in range(len(self.lamp_schedule)):
                    if self.lamp_schedule.at[i, 'UV'] == 10:
                        SP_timings.append(self.lamp_schedule.loc[:i, 'Time(s)'].sum()-1)
    
                seconds_to_remove = []
                for i in SP_timings:
                    values = np.arange(i, i + 90, step = 1)
                    seconds_to_remove.append(values)
                    
                self.gas_exchange_data_no_SP = self.gas_exchange_data[np.isin(self.gas_exchange_data.seconds, seconds_to_remove, invert=True)]
    
                return True
            else:
                print(f'flashes cannot be removed for {self.plantID} because either the lamp schedule or the gas exchange data don\'t exist')
                return False
        else:
            return True
    
    def create_data_inspection_plots(self):
        #ST measurement
        fig, ax = plt.subplots(1,3, figsize=(18, 5))


        ax[0].fill_between([0,30], 0, 1.2, color='darkred', alpha=0.4, label='')
        ax[0].fill_between([30,46], 0, 1.2, color='lightcoral', alpha=0.4, label='')
        ax[0].fill_between([46,62], 0, 1.2, color='darkred', alpha=0.4, label='')
        ax[0].fill_between([62,78], 0, 1.2, color='lightcoral', alpha=0.4, label='')

        Fluo, = ax[0].plot(self.dual_pam_data.minutes, self.dual_pam_data['Fm_prime']/self.Fm, c='b', label='Fm\'/Fm', marker='o', ls="")
        if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
            ax[0].set_xlim(min(self.dual_pam_data.minutes), 88)
        else:
            ax[0].set_xlim(min(self.dual_pam_data.minutes), 78)

        ax[0].set_ylim(min(self.dual_pam_data['Fm_prime']/self.Fm) -0.05,
                       max(self.dual_pam_data['Fm_prime']/self.Fm)+0.05)


        ax1 = ax[0].twinx()
        P700, = ax1.plot(self.dual_pam_data.minutes, (self.dual_pam_data.Phigh-self.dual_pam_data.Ps)/self.Pm, c='orange', label='QY of PSI', marker='o', ls="")
        ax1.set_ylim(0, 1.2)
        
        par_string = 'PAR: {:.2f} \nFR: {:.2f}'.format(find_sum_between_bounds_prestandardized(self.WL_spectrum, [400,701]), find_sum_between_bounds_prestandardized(self.WL_spectrum, [700,751]))
        par_fr_string = 'PAR: {:.2f} \nFR: {:.2f}'.format(find_sum_between_bounds_prestandardized(self.WL_FR_spectrum, [400,701]), find_sum_between_bounds_prestandardized(self.WL_FR_spectrum, [700,751]))
        ax[0].text(0.61, 0.9, par_fr_string, fontsize=7, transform=ax[0].transAxes)    
        ax[0].text(0.83, 0.9, par_string, fontsize=7, transform=ax[0].transAxes)


        ax[0].text(0.10, 0.02, 'PAR + FR', transform=ax[0].transAxes)    
        ax[0].text(0.40, 0.02, 'PAR', transform=ax[0].transAxes)
        ax[0].text(0.59, 0.02, 'PAR + FR', transform=ax[0].transAxes)
        ax[0].text(0.85, 0.02, 'PAR', transform=ax[0].transAxes)


        ax[0].set_xlabel('Time (min)')
        ax[0].set_ylabel('Fm')
        ax1.set_ylabel('QY of PSI')
        ax[0].set_title(f'dual PAM measurements: {self.plantID}')
        ax[0].legend(handles=[Fluo, P700], loc='upper left')

        #_________________________________________________________________________________
        ax[1].fill_between([0,30], -200, 100, color='darkred', alpha=0.4, label='')
        ax[1].fill_between([30,46], -10, 100, color='lightcoral', alpha=0.4, label='')
        ax[1].fill_between([46,62], -10, 10, color='darkred', alpha=0.4, label='')
        ax[1].fill_between([62,78], -10, 10, color='lightcoral', alpha=0.4, label='')

        Photo, = ax[1].plot(self.gas_exchange_data_no_SP.minutes, self.gas_exchange_data_no_SP['A'], c='b', label='CO2', marker='o', markersize=2, ls="")
        #Photo, = ax[1].plot(self.gas_exchange_data.minutes, self.gas_exchange_data['A'], c='b', label='CO2', marker='o', markersize=2, ls="")
        max_A = max(self.gas_exchange_data_no_SP[self.gas_exchange_data_no_SP.minutes<78].A)
        if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
            ax[1].set_xlim(0, 88)
        else:
            ax[1].set_xlim(0, 78)
        ax[1].set_ylim(max_A-0.7, max_A+0.05)

        ax1 = ax[1].twinx()
        gsw, = ax1.plot(self.gas_exchange_data_no_SP.minutes, self.gas_exchange_data_no_SP['gsw'], c='orange', label='gsw', marker='o', markersize=2, ls="")
        ax1.set_ylim(0,max(self.gas_exchange_data_no_SP['gsw'])+0.05)


        ax[1].text(0.61, 0.9, par_fr_string, fontsize=7, transform=ax[1].transAxes)    
        ax[1].text(0.83, 0.9, par_string, fontsize=7, transform=ax[1].transAxes)

        ax[1].text(0.10, 0.02, 'PAR + FR', transform=ax[1].transAxes)    
        ax[1].text(0.40, 0.02, 'PAR', transform=ax[1].transAxes)
        ax[1].text(0.59, 0.02, 'PAR + FR', transform=ax[1].transAxes)
        ax[1].text(0.85, 0.02, 'PAR', transform=ax[1].transAxes)


        ax[1].set_xlabel('Time (min)')
        ax[1].set_ylabel(r'$CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')
        ax1.set_ylabel('gsw')
        ax[1].set_title(f'gas exchange ST measurement: {self.plantID}')
        ax[1].legend(handles=[Photo, gsw], loc='upper left')

        #______________
    
        ax[2].fill_between([0,30], -10, 20, color='darkred', alpha=0.4)
        ax[2].fill_between([30,46], -10, 20, color='lightcoral', alpha=0.4)
        ax[2].fill_between([46,62], -10, 20, color='darkred', alpha=0.4)
        ax[2].fill_between([62,78], -10, 20, color='lightcoral', alpha=0.4)
        


        if self.plantID[0:2] == 'SF':
            qT, = ax[2].plot(self.dual_pam_data.minutes, 
                             (self.dual_pam_data.at[21, 'Fm_prime'] - self.dual_pam_data['Fm_prime']) / self.dual_pam_data.at[21, 'Fm_prime'] *100,
                             c='orange', label='qT', marker='o', ls="")
        elif self.plantID == 'NT8' or self.plantID == 'NT9':
            qT, = ax[2].plot(self.dual_pam_data.minutes, 
                             (self.dual_pam_data.at[27, 'Fm_prime'] - self.dual_pam_data['Fm_prime']) / self.dual_pam_data.at[27, 'Fm_prime'] *100,
                             c='orange', label='qT', marker='o', ls="")
        elif self.plantID[0:2] != 'SF':
            qT, = ax[2].plot(self.dual_pam_data.minutes, 
                             (self.dual_pam_data.at[25, 'Fm_prime'] - self.dual_pam_data['Fm_prime']) / self.dual_pam_data.at[25, 'Fm_prime'] *100,
                             c='orange', label='qT', marker='o', ls="")
        if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
            ax[2].set_xlim(40,88)
        else:
            ax[2].set_xlim(30,78)
        ax[2].set_ylim(-2,15)

                
        ax[2].text(0.10, 0.02, 'PAR', transform=ax[2].transAxes)
        ax[2].text(0.45, 0.02, 'PAR + FR', transform=ax[2].transAxes)
        ax[2].text(0.75, 0.02, 'PAR', transform=ax[2].transAxes)


        ax[2].set_xlabel('Time (min)')
        ax[2].set_ylabel('qT (%)')
        ax[2].set_title('qT')
        ax[2].legend(handles=[qT], loc='upper left')

        fig.tight_layout()
        analysis_folder = find_or_create_results_folder(self.plantID)
        save_name = os.path.join(analysis_folder,'ST_measurement_overview.png')
        fig.savefig(save_name,dpi=400)
        plt.close()     
        
        if True:
            fig, ax = plt.subplots(1,1, figsize=(5,5))

            Photo_2, = ax.plot(self.gas_exchange_data_no_SP.minutes, self.gas_exchange_data_no_SP['A']-0.05, c='orange', label='noflash', marker='o', markersize=2, ls="")
            Photo, = ax.plot(self.gas_exchange_data.minutes, self.gas_exchange_data['A'], c='b', label='flash', marker='o', markersize=2, ls="")
            max_A = max(self.gas_exchange_data_no_SP[self.gas_exchange_data_no_SP.minutes<78].A)
            ax.set_xlim(30,46)
            ax.set_ylim(max_A-0.7, max_A+0.05)
    


    
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(r'$CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')

            ax.set_title(f'gas exchange ST measurement: {self.plantID}')
            ax.legend(handles=[Photo, Photo_2], loc='upper left')
            fig.tight_layout()
            analysis_folder = find_or_create_results_folder(self.plantID)
            save_name = os.path.join(analysis_folder,f'{self.plantID} flash removal check.png')
            fig.savefig(save_name,dpi=400)
            plt.close()   
        return None
    
    def analyse_R_day_measurement(self, update=False):
        if update or (not hasattr(self, 'R_day')):
            if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
                start_R_day = 88
            else:
                start_R_day = 78
            gas_frame_R_day = self.gas_exchange_data[self.gas_exchange_data.minutes > start_R_day]
            analysis_time = start_R_day+5.5
            
            start_index_R_day_light = int(self.incident_spectra.index[(self.incident_spectra.PAR == 3) & (self.incident_spectra.FR == 0)].tolist()[0])
    
            PAR = self.incident_spectra.loc[start_index_R_day_light:start_index_R_day_light+7, 'chamber_PAR']
            assimilation = []
            for i in range(len(PAR)):
                analysis_frame = gas_frame_R_day[(gas_frame_R_day.minutes > analysis_time) & (gas_frame_R_day.minutes < analysis_time+0.4)]
    
                assimilation.append(analysis_frame.A.mean())
                analysis_time +=6
            
            R_day_frame = pd.DataFrame(data = {'PAR': PAR,
                                               'assimilation': assimilation})
    
    
                
            fig, ax = plt.subplots(1,2, figsize=(10, 5))
            Photo, = ax[0].plot((gas_frame_R_day.minutes - start_R_day), gas_frame_R_day['A'], c='b', label='CO2', marker='', ls="-")
            
    
            ax[0].set_xlim(0, 42)
            ax[0].set_ylim(min(gas_frame_R_day.A)-0.05, max(gas_frame_R_day.A)+0.05)
    
            ax[0].set_xlabel('Time (min)')
            ax[0].set_ylabel(r'$CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')
    
            ax[0].legend(handles=[Photo], loc='upper right')
            
            
            cutoff_for_R_day = 5
            coef = np.polyfit(R_day_frame.iloc[:cutoff_for_R_day,0],R_day_frame.iloc[:cutoff_for_R_day,1],1)
            poly1d_fn = np.poly1d(coef) 
            
            ax[1].plot(np.arange(0, 200),poly1d_fn(np.arange(0,200)),'--k')
            
            
            ax[1].plot(R_day_frame.iloc[:cutoff_for_R_day,0],R_day_frame.iloc[:cutoff_for_R_day,1],c='green', marker='o', ls='')
            ax[1].plot(R_day_frame.iloc[cutoff_for_R_day:,0],R_day_frame.iloc[cutoff_for_R_day:,1],c='red', marker='o', ls='')
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            string = r'$R_{day}$ = ' + str(round(-coef[1], 3)) + ' \n' + r'$QY_{CO2}$ = ' + str(round(coef[0], 3))
            
            ax[1].text(0.60, 0.1, string, transform=ax[1].transAxes, fontsize=11, bbox=props)
              
            ax[1].set_xlabel(r'PAR intensity [$\mu$mol $m^{-2}$ $s^{-1}$]')
            ax[1].set_ylabel(r'$CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')
            ax[1].set_xlim(-5,210)
            ax[1].set_ylim(-2,9.5)
            fig.tight_layout()
            analysis_folder = find_or_create_results_folder(self.plantID)
            save_name = os.path.join(analysis_folder,'R_day_determination.png')
            fig.savefig(save_name,dpi=400)
            plt.close()  
            
            self.R_day_data = R_day_frame
            self.R_day = coef[1]
            self.QY_CO2 = coef[0]
            return None
        else:
            return True
        
    def create_poster_plot(self):
        print('Watch out create poster plot uses wrong baseline value for qT')
        #ST measurement
        fig, ax = plt.subplots(1,2, figsize=(12, 5))


        ax[0].fill_between([0,30], 0, 1.2, color='darkred', alpha=0.4, label='')
        ax[0].fill_between([30,46], 0, 1.2, color='lightcoral', alpha=0.4, label='')
        ax[0].fill_between([46,62], 0, 1.2, color='darkred', alpha=0.4, label='')
        ax[0].fill_between([62,78], 0, 1.2, color='lightcoral', alpha=0.4, label='')

        P700, = ax[0].plot(self.dual_pam_data.minutes, (self.dual_pam_data.Phigh-self.dual_pam_data.Ps)/self.Pm, c='orange', label='QY of PSI', marker='s', ls="")

        if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
            ax[0].set_xlim(40, 88)
        else:
            ax[0].set_xlim(30, 78)

        ax[0].set_ylim(0,1.2)
        


        ax1 = ax[0].twinx()
        if self.plantID[0:2] != 'SF':
            qT, = ax1.plot(self.dual_pam_data.minutes, 
                             (self.dual_pam_data.at[23, 'Fm_prime'] - self.dual_pam_data['Fm_prime']) / self.dual_pam_data.at[23, 'Fm_prime'] *100,
                             c='orange', label='qT', marker='o', ls="")
        elif self.plantID[0:2] == 'SF':
            qT, = ax1.plot(self.dual_pam_data.minutes, 
                             (self.dual_pam_data.at[21, 'Fm_prime'] - self.dual_pam_data['Fm_prime']) / self.dual_pam_data.at[21, 'Fm_prime'] *100,
                             c='blue', label='qT', marker='o', ls="")
        ax1.set_ylim(-1, 8)
        
        par_string = 'WL: {:.2f} \nFR: {:.2f}'.format(find_sum_between_bounds_prestandardized(self.WL_spectrum, [400,701]), find_sum_between_bounds_prestandardized(self.WL_spectrum, [700,751]))
        par_fr_string = 'WL: {:.2f} \nFR: {:.2f}'.format(find_sum_between_bounds_prestandardized(self.WL_FR_spectrum, [400,701]), find_sum_between_bounds_prestandardized(self.WL_FR_spectrum, [700,751]))
        ax[0].text(1/6, 0.89, par_string, horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)
        ax[0].text(1/2, 0.89, par_fr_string, horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)    
        ax[0].text(5/6, 0.89, par_string, horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)


        ax[0].text(1/6, 0.96, 'Sun', horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)
        ax[0].text(1/2, 0.96, 'Shade', horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)
        ax[0].text(5/6, 0.96, 'Sun', horizontalalignment='center', fontsize=9, transform=ax[0].transAxes)


        ax[0].set_xlabel('Time (min)')
        ax[0].set_ylabel('QY of PSI')
        ax1.set_ylabel('qT (%)')
        ax[0].set_title('Dual PAM measurement')
        ax[0].legend(handles=[P700, qT], loc='lower left')
        
        
        #_________________________________________________________________________________
        ax[1].fill_between([0,30], -200, 100, color='darkred', alpha=0.4, label='')
        ax[1].fill_between([30,46], -10, 100, color='lightcoral', alpha=0.4, label='')
        ax[1].fill_between([46,62], -10, 10, color='darkred', alpha=0.4, label='')
        ax[1].fill_between([62,78], -10, 10, color='lightcoral', alpha=0.4, label='')

        Photo, = ax[1].plot(self.gas_exchange_data_no_SP.minutes, self.gas_exchange_data_no_SP['A'], c='b', label=r'CO$_2$', marker='o', markersize=3, ls="")
        #Photo, = ax[1].plot(self.gas_exchange_data.minutes, self.gas_exchange_data['A'], c='b', label='CO2', marker='o', markersize=2, ls="")
        max_A = max(self.gas_exchange_data_no_SP[self.gas_exchange_data_no_SP.minutes<78].A)
        if (self.plantID=='NT8')^(self.plantID=='NT9')^(self.plantID=='SF1'):
            ax[1].set_xlim(40, 88)
        else:
            ax[1].set_xlim(30, 78)
        ax[1].set_ylim(3.65,4)

        ax1 = ax[1].twinx()
        gsw, = ax1.plot(self.gas_exchange_data_no_SP.minutes, self.gas_exchange_data_no_SP['gsw'], c='orange', label='gsw', marker='o', markersize=3, ls="")
        ax1.set_ylim(0.1,0.3)

        ax[1].text(1/6, 0.89, par_string, horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)
        ax[1].text(1/2, 0.89, par_fr_string, horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)    
        ax[1].text(5/6, 0.89, par_string, horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)


        ax[1].text(1/6, 0.96, 'Sun', horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)
        ax[1].text(1/2, 0.96, 'Shade', horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)
        ax[1].text(5/6, 0.96, 'Sun', horizontalalignment='center', fontsize=9, transform=ax[1].transAxes)


        ax[1].set_xlabel('Time (min)')
        ax[1].set_ylabel(r'$CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')
        ax1.set_ylabel(r'$g_{sw}$')
        ax[1].set_title('Gas exchange measurement')
        ax[1].legend(handles=[Photo, gsw], loc='lower left')


        fig.tight_layout()
        analysis_folder = find_or_create_results_folder(self.plantID)
        save_name = os.path.join(analysis_folder,'poster_image_ST.png')
        fig.savefig(save_name,dpi=400)
        plt.close()     
        return True
    
#%% Starts analysis with the data
if __name__ == "__main__":
    plant_instances = pd.read_excel('measurement_overview.xlsx',converters={'licor_start_time':str,'lamp_start_time':str})
    plant_instances[["date", "licor_start_time", "lamp_start_time", 'sow date']] = plant_instances[["date", "licor_start_time", "lamp_start_time", 'sow date']].astype(str)
    #removed first 3 measurements as data is poor, and breaks the analysis (light correction at 2/2 through hole is missing)
    # Final two are also removed as they are measurements for technical replicates
    plant_instances = plant_instances[3:-2]
    plant_instances = plant_instances[plant_instances.plantID == 'Ph3']
    plant_instances.set_index('plantID', inplace=True)
    
    
    test_plants = np.array(plant_instances.index)

    for test_plant in test_plants:

        if os.path.isfile(f'./pickles/{test_plant}.pkl'):
            print(f'loaded {test_plant} data from pickle')
            with open(f'pickles/{test_plant}.pkl', 'rb') as file:
                plant_test = pkl.load(file)
        else:
            print(f'created new object for {test_plant}')
            plant_test = plant_instance(test_plant, plant_instances.at[test_plant, 'date'],
                                        plant_instances.at[test_plant, 'licor_start_time'],
                                        plant_instances.at[test_plant, 'lamp_start_time'])
        
        update_Aci = False
        update_gas_ex = False
        update_dual_pam = False 
        #selects which things to update
        #if test_plant == 'NT5':
        #    update_Aci=True
        #    update_gas_ex = True
    
        plant_test.obtain_gas_data(update=update_gas_ex)
        plant_test.obtain_lamp_script(update=False)
        plant_test.remove_flashes_from_gas_exchange_data(update=update_gas_ex)
        plant_test.obtain_dual_pam_files(plot_figures=True,update=update_dual_pam)
        plant_test.obtain_Aci_data(plot_figures=True,update=update_Aci)
        plant_test.load_and_calculate_chl_content(plant_instances.at[test_plant, 'chl_vol_ml'],
                                                  plant_instances.at[test_plant, 'chl_dil'],update=False)
        plant_test.load_and_check_light_spread_chamber(plot_figures=True,update=False)
        plant_test.load_leaf_absorbance(plot_figures=True,update=False)
        plant_test.obtain_incident_spectra(plot_figures=True,update=False)
        plant_test.analyse_R_day_measurement(update=update_gas_ex)
        plant_test.create_data_inspection_plots()
        plant_test.load_and_analyse_CB6F_rates(update=False)
        plant_test.load_streak_measurements(update=True)
        plant_test.analyse_k_NPQ(update=False)
        if False:
            plant_test.create_poster_plot()
        with open(f'pickles/{test_plant}.pkl', 'wb') as file:
            pkl.dump(plant_test, file)
        #print(f'Plant {plant_test.plantID} has Fv/Fm: {plant_test.Fv_Fm} ')








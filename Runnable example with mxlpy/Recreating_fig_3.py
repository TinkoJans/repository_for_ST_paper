

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:37:03 2026

@author: jans008
"""

#%% Initialising file
#loading all the required packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle as pkl
import xarray as xr
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

import ST_model as ST

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

#%% Initialises a wide range of supporting parameters
Chl_per_area = 0.00042 # mol m-2 
PS_per_chl = 0.0046 # mol/mol chl, Ibrahim 2022,
PSI_PSII_ratio = 0.6 #PSI/PSII, McKenzie 2020
PQ_per_chl = 0.01 # mol/mol chl, kruk 2006
concentration_CB6F = 0.00076 # mol/mol Chl, ibrahim 2022
k_CB6F_star = 300 # mol PQH2 mol-1 sites s-1, catalytic constant for Cyt B6F assuming relaxed photosynthetic control

concentration_PSII = PS_per_chl * (1 - PSI_PSII_ratio / (1 + PSI_PSII_ratio)) # mol/mol Chl
concentration_PSI = PS_per_chl - concentration_PSII # mol/mol Chl

K_P1 = 16 # ns-1, rate of photochemistry
K_D1 = 0.55 # ns-1, rate of constitutive heat loss
K_F1 = 0.05 # ns-1, rate of fluorescence
K_X1 = K_P1 # ns-1, rate of dissipation in closed P700, set equal to K_P1


#load spectrum of the PAM
PAM_spect = np.loadtxt('model parameters FvCBwd model/Dual_pam_measuring_light_spectrum.txt', delimiter='\t', skiprows=1)[:371,1]
PAM_spect[:50] = 0
Ci = 280*1e-6

#%%import leaf absorbance and absorbance by photosynthetic pigments. 
parameters = pd.read_excel('model parameters fvCBwd model/parameters.xlsx', header=2)
# Inter- and extrapolates the missing values in the absorption by non-photosynthetic pigmens.  
known_abs_pig = np.transpose(np.array(parameters[parameters['alpha_pigments'].notna()][['Wavelength(nm)', 'alpha_pigments']]))
x_new = np.array(parameters['Wavelength(nm)'])
parameters['alpha_pigments'] = np.interp(x_new, known_abs_pig[0], known_abs_pig[1])
leaf_absorbance = np.array(parameters.leaf_absorbance)
absorbance_photopigments = np.array(parameters.alpha_pigments)

#import absorbance spectra of protein components.
file_names = ['Hogewoning_PSI', 'Hogewoning_PSII_core', 'Hogewoning_LHCII']
spectra = np.zeros((3,371))
for i in range(len(file_names)):
    file_name = 'model parameters FvCBwd model/' + file_names[i] + '.csv'
    file = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
    spectrum = np.array(file.intensity)
    spectra[i] = spectrum
    
# Number of chlorophylls per protein complex.
PS_table = pd.DataFrame({'chl_a': [148, 108, 24],
                         'chl_b': [12, 49, 18],
                         'chl_tot': [160, 157, 42]}, index=['PSI-LHCI', 'CSM', 'LHCII'])
#PSI-LHCI from Pan 2018; CSM from Su 2014; LHCII from Laisk 2014 


PSI_LHCI_chl = PS_table.at['PSI-LHCI', 'chl_a'] + 0.7 * PS_table.at['PSI-LHCI', 'chl_b']
spectrum_PSI_LHCI = (PSI_LHCI_chl / spectra[0][250:].sum(axis=0)) * (spectra[0])

CSM_chl = PS_table.at['CSM', 'chl_a'] + 0.7 * PS_table.at['CSM', 'chl_b']
spectrum_CSM = (CSM_chl / spectra[1][250:].sum(axis=0)) * (spectra[1])

LHCII_chl = PS_table.at['LHCII', 'chl_a'] + 0.7 * PS_table.at['LHCII', 'chl_b']
spectrum_LHCII = (LHCII_chl / spectra[2][250:].sum(axis=0)) * (spectra[2])


#%%Create parameter dictionary
params = {'chloroplastic_oxygen' : 212.7*1e-3, # bar, 21% of atmospheric pressure
          'Vcmax' : 100 *1e-6, #umol m-2 s-1, generic value
          'Rd': 1 * 1e-6, #umol m-2 s-1, generic value
          'gamma_star' : 40*1e-6, # bar at 25C, standard value from Duursma 2015 and Bernacchi 2001  
          'MM_constant_oxy' : 175*1e-3, # bar at 25C,  Duursma 2015 and Bernacchi 2001
          'MM_constant_carboxy' : 300*1e-6, # bar at 25C, Duursma 2015 and Bernacchi 2001
          'g_meso' : 0.3, #mol m-2 s-1 bar-1 
          'Ci' : Ci,
          
          
          'CB6F_recovery_rate':k_CB6F_star * concentration_CB6F * Chl_per_area,
          'eta_LET' : 0.75, # mol ATP mol-1 e-1
          'eta_CET' : 1, # mol ATP mol-1 e-1
          
          'D_P680_tot' : concentration_PSII * Chl_per_area, # mol m-2
          'D_P700_tot' : concentration_PSI * Chl_per_area, # mol m-2
          'PQ_pool' : PQ_per_chl * Chl_per_area, # mol m-2
          

          'phosphorylation_rate_constant' : 0.2/60, #Fitted phosphorylation rate in Jans2026
          'dephosphorylation_rate_constant' : 0.5/60, #Fitted dephosphorylation rate in Jans2026
          
          #Photosystem dissipation rate constants 
          'K_P2' : 3.3, # ns-1, rate of photochemistry
          'K_D2' : 0.55, # ns-1, rate of constitutive heat loss
          'K_F2' : 0.05, # ns-1, rate of fluorescence
          'K_U2' : 2.1, # ns-1, to do
          'K_N2' : 0, #ns-1
          
          'K_P1' : K_P1,
          'K_F1' : K_F1,
          'rate_tot_P700_open' : K_P1 + K_D1 + K_F1,
          'rate_tot_P700_closed' : K_X1 + K_D1 + K_F1,
        

          #measured spectrum of the Dual-PAM fluorescence measuring light 
          'PAM_spect' : PAM_spect,
          'epsilon_F1' : 2, #Rough values from STREAK measurements
          'epsilon_F2' : 1,
          
          #set all the spectra
          'spectrum_LHCII' : spectrum_LHCII, 
          'spectrum_PSI_LHCI' : spectrum_PSI_LHCI, 
          'spectrum_CSM' : spectrum_CSM,
          'leaf_absorbance' : leaf_absorbance, 
          'absorbance_photopigments' : absorbance_photopigments,
          }




#%% Creates the 2D light list of alternating 480 and 700 nm LEDs

#Loads incident light spectra Taylor 2019 
#load light spectra from csv file and calculate spectrum list from 380 to 730 nm.   
def load_spectrum(lamp_name,  light_intensity):  
    #light_intensity in umol m-2 s-1
    file_name = 'model parameters FvCBwd model/'+ lamp_name + '.csv'
    spectrum_df = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
    spectrum = np.zeros(371)
    if spectrum_df.wavelength[0] < 380:
        print('Watch out, spectrum provided have wavelengths lower than 380, this will lead to errors')
    else: 
        offset = int(spectrum_df.wavelength[0]-380)
        max_index = min(len(spectrum_df.wavelength), 370-offset)
        spectrum[offset:offset+max_index] = spectrum_df.intensity[0:max_index]
        
        #normalises spectrum to have area=1
        spectrum = spectrum /spectrum.sum()
 
        #normalises spectrum to have area= absorbed light intensity
        spectrum = light_intensity*1e-6 *spectrum
    return spectrum

def create_light_list(light_intensity, time_steps_model = 5):
    
    correction_factor = 55/40 # correction factor is to get approximately equal absorbed irradiances
    spectrum_480 = load_spectrum('480_light', light_intensity) #Correction factor is applied to get 40 umol absorbed photons
    spectrum_700 = load_spectrum('700_light', light_intensity * correction_factor) #Correction factor is applied to get 40 umol absorbed photons        

    #initialise light and simulation time

    section_minutes = 15
    section_time = section_minutes * 60

    model_length =int( 4 * section_time / time_steps_model)
    Ci_model = np.repeat([Ci], model_length)
    Time_min = np.arange(0,4 * section_time/60, time_steps_model/60)
    

    light_list = np.zeros((model_length,371))
    for i in range(len(light_list)):
        if (i>=len(light_list)/4) and (i<len(light_list)*2/4) or (i>len(light_list) * 3/4):
            light_list[i] = spectrum_700
        else:
            light_list[i] = spectrum_480 
            
    return light_list, Ci_model, Time_min



#%%
if __name__ == '__main__':
    
    
    
    #%% Plot equivalent to Fig 3
    light_intensity = 40
    time_steps_model = 1
    light_list, Ci_list, Time_min = create_light_list(light_intensity, time_steps_model)
    Time_sec = Time_min * 60

    
    correction_factor = 55/40 # correction factor is to get approximately equal absorbed irradiances
    spectrum_480 = load_spectrum('480_light', light_intensity) #Correction factor is applied to get 40 umol absorbed photons
    spectrum_700 = load_spectrum('700_light', light_intensity * correction_factor) #Correction factor is applied to get 40 umol absorbed photons        
    
    ST.set_spectra(
        spectrum_0_=spectrum_480, spectrum_1_=spectrum_700,
        spectrum_LHCII_=spectrum_LHCII, spectrum_PSI_LHCI_=spectrum_PSI_LHCI,
        spectrum_CSM_=spectrum_CSM, leaf_absorbance_=leaf_absorbance,
        PAM_spect_=PAM_spect, absorbance_photopigments_=absorbance_photopigments,
    )
    
    model = ST.build_model(params)  # your existing params dict
    protocol = ST.build_protocol(section_minutes=[15,15,15,15],
                                 section_light_ids= [0.0,1.0,0.0,1.0])
    sim = ST.make_simulator(model)
    variables, fluxes = (sim.simulate_protocol_time_course(protocol, time_points=np.arange(0,3600,10))
                         .get_result().unwrap_or_err())
    sim.clear_results()   
    variables['Time_min'] = variables.index/60

    

    fontsize=12
    colors = ['blue', 'orange', 'red']
    time_offset =30
    fig, axs = plt.subplots(2,3, figsize =(12,8))
    axs[0,0].plot(variables['Time_min']-time_offset, variables['P680_excitation']*1e6, ls='-', color='black', label="PSII excitation")
    axs[0,0].plot(variables['Time_min']-time_offset, variables['P700_excitation']*1e6, ls='--', color='black', label="PSI excitation")
    axs[0,0].set_ylabel(r'Photosystem excitation $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[0,0].set_ylim(0, variables['P700_excitation'].max()*1e6+5)
    axs[0,0].legend(fontsize=fontsize)
    
    
    P680_plot, = axs[0,1].plot(variables['Time_min']-time_offset, variables['D_P680'], ls='-', color='black', label="P680")
    P700_plot, = axs[0,1].plot(variables['Time_min']-time_offset, variables['D_P700'], ls='--', color='black', label="P700")
    PQ_ax = axs[0,1].twinx()
    PQ_plot, = PQ_ax.plot(variables['Time_min']-time_offset, variables['PQH2'], color='orange', ls='-', label=r"$PQH_2$/$PQ_{tot}$")
    axs[0,1].set_ylabel('Fraction open reaction centres', fontsize=fontsize)
    PQ_ax.set_ylabel(r'$PQH_2/PQ_{tot}$', fontsize=fontsize)
    axs[0,1].set_ylim(-0.22, 1.1)
    PQ_ax.set_ylim(-0.22, 1.1)
    axs[0,1].legend(handles=[P680_plot, P700_plot, PQ_plot], loc='lower right', fontsize=fontsize)
    
    
    QY = variables['D_P700']
    axs[0,2].plot(variables['Time_min']-time_offset,1-QY , ls='--',c='orange', label='P700 donor limitation')
    axs[0,2].plot(variables['Time_min']-time_offset, QY, ls='--', c='black', label='P700 photochemistry')
    axs[0,2].set_ylabel('QY', fontsize=fontsize)
    axs[0,2].legend(loc='lower right', fontsize=fontsize)
    axs[0,2].set_ylim(-0.25,1.1)
    
    
    axs[1,0].plot(variables['Time_min']-time_offset, variables['LHCII_per_PSII'], ls='-', color='black', label="LHCII per PSII")
    axs[1,0].plot(variables['Time_min']-time_offset, variables['LHCII_per_PSI'], ls='--', color='black', label="LHCII per PSI")
    axs[1,0].set_ylabel('LHCII per PS', fontsize=fontsize)
    axs[1,0].set_ylim(0,variables['LHCII_per_PSII'].max()+0.4)
    axs[1,0].legend(loc='lower left', fontsize=fontsize)
    
    
    axs[1,1].plot(variables['Time_min']-time_offset, variables['J_P680_forward']*1e6, ls='-', color='black', label="PSII electron flow ")
    axs[1,1].plot(variables['Time_min']-time_offset, variables['J_P700_forward']*1e6, ls='--', color='black', label="PSI electron flow")
    axs[1,1].set_ylabel(r'Electron flow rate $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[1,1].set_ylim(0,variables['J_P700_forward'].max()*1e6+5)
    axs[1,1].legend(loc='lower right', fontsize=fontsize)
    
    qT = (max(variables['Fm'])-variables['Fm'])/max(variables['Fm']) *100
    A_plot, = axs[1,2].plot(variables['Time_min']-time_offset,  variables['A']*1e6, color='orange', label='Net Assimilation')
    axs[1,2].set_ylabel(r'Net Assimilation  $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[1,2].set_ylim(0,max( variables['A']*1e6)+0.4)
    qT_ax = axs[1,2].twinx()
    qT_plot, = qT_ax.plot(variables['Time_min']-time_offset, qT, color='blue', ls='-', label='qT')
    qT_ax.set_ylabel('qT (%)', fontsize=fontsize)
    qT_ax.set_ylim(0,12)
    axs[1,2].legend(handles = [A_plot, qT_plot], loc='lower left', fontsize=fontsize)
    

    
    
    
    colors = ['lightskyblue', 'lightcoral']
    texts = ['Blue LED', 'FR LED']
    for i in range(2):
        for j in range(3):
            axs[i,j].tick_params(axis='both', which='major', labelsize=fontsize)
            axs[i,j].set_xlim(0,30)
            axs[1,j].set_xlabel('Time (min)')
            for k in range(2):
                
                banner = mpatches.Rectangle((0+k*0.5, 0.94), 0.5+k*0.5, 0.06,
                                            transform=axs[i,j].transAxes,  # axes coords
                                            facecolor=colors[k],
                                            edgecolor=colors[k], linewidth=0.8)
                axs[i,j].add_patch(banner)
                axs[i,j].text(0.25+k*0.5, 0.975, texts[k], transform=axs[i,j].transAxes,
                        va="center_baseline", ha="center", fontsize=12, color="black")
                axs[i,j].vlines(15,-10,150, ls='--', color='lightgrey', zorder=0)
    axs[1,1].set_yticks([0,5,10,15,20])
    axs[1,2].set_yticks([0,0.5,1,1.5,2])
    
    fig.text(0.01,0.97, 'A', fontsize=15)
    fig.text(0.33,0.97, 'B', fontsize=15)
    fig.text(0.66,0.97, 'C', fontsize=15)
    
    fig.text(0.01,0.49, 'D', fontsize=15)
    fig.text(0.33,0.49, 'E', fontsize=15)
    fig.text(0.66,0.49, 'F', fontsize=15)
    
            

    fig.tight_layout()
    plt.show()
    plt.close()
    

    
    
    
    
    
    
    
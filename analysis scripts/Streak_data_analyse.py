# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:20:15 2025

@author: jans008
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to handle button click
def open_ascii_files(folder_path):
    # Dictionary to store NumPy arrays with filenames as keys
    array_DAS = {} 
    # Loop through all files in the selected folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .ascii extension (ASCII file)
        if filename.endswith(".ascii"):
            # Extract the NumPy array from the ASCII file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                # Skip the first row (header) in the ASCII file
                lines = file.readlines()[1:]
                data = np.array([list(map(float, line.split())) for line in lines])
                # Get the indices of columns with same content as the first column
                duplicate_indices = [i for i in range(1, data.shape[1]) if np.array_equal(data[:, 0], data[:, i])]
                # Create a new array without the duplicate columns
                unique_indices = list(set(range(data.shape[1])) - set(duplicate_indices))
                new_data = data[:, unique_indices]

                if 'DAS' in filename: 
                    max_PSII_short=max(new_data[:,2])
                    new_data[:,1:]=new_data[:,1:]/max_PSII_short
                    array_DAS[filename[0:2]] = new_data
                if 'LHCII' in filename:
                    array_DAS['LHCII'] = new_data
    # Return the dictionary of NumPy arrays
    return array_DAS

def open_lifetime_files(folder_path):
    #F0
    array_lifetime = {}
    filename = os.path.join(folder_path, 'F0_k')
    with open(filename, 'r') as data_file:
        lines = data_file.readlines()[6:8]
        lifetimes = [float(lifetime[5:-1]) for lifetime in lines]
        array_lifetime['F0'] = [lifetimes[0], lifetimes[0], lifetimes[1]]
        #Fm
    filename = os.path.join(folder_path, 'Fm_k')
    with open(filename, 'r') as data_file:
        lines = data_file.readlines()[7:10]
        lifetimes = [float(lifetime[5:-1]) for lifetime in lines]
        array_lifetime['Fm'] = lifetimes
    return array_lifetime

def norm_das_dict_to_area(DAS_dict):
    DAS_dict_norm = {}
    for key in DAS_dict:
        norm_array = DAS_dict[key] /DAS_dict[key][:,1:].sum()
        DAS_dict_norm[key] = norm_array
    return DAS_dict_norm


def lin_interpolate_spectrum(ref_wl, ref_spect, target_wls):
    int_spect = []
    for target_wl in target_wls:
        index = [ n for n,i in enumerate(ref_wl) if i>target_wl ][0]
        new_value = ref_spect[index-1] + (ref_spect[index] - ref_spect[index-1])/(ref_wl[index]-ref_wl[index-1]) * (target_wl-ref_wl[index-1])
        int_spect.append(new_value)
    return np.array(int_spect)

def get_LHCII_correction_spectrum(target_wavelengths):
    DAS = open_ascii_files('Streak_measurements/Results')
    wls_LHCII = DAS['LHCII'][:,0]
    spect_LHCII = DAS['LHCII'][:,1]
    spect = lin_interpolate_spectrum(wls_LHCII, spect_LHCII, target_wavelengths)
    
    return spect
#%%
def analyse_streak_data(folder_location, plantID):
    dict_DAS = open_ascii_files(folder_location)
    dict_time = open_lifetime_files(folder_location)

    #normalise DAS to total area 1
    dict_DAS_norm = norm_das_dict_to_area(dict_DAS)

    #unmixes PSII contribution to PSI spectrum in F0. 
    #it assumes PSI spectrum is the same in Fm as in F0
    unmixed_DAS = {}
    PSI_F0 = dict_DAS_norm['F0'][:,1]
    PSI_Fm = dict_DAS_norm['Fm'][:,1]
    PSI_diff = PSI_F0-PSI_Fm
    
    unmixed_DAS['F0'] = np.stack((dict_DAS_norm['F0'][:,0],PSI_Fm,PSI_diff,dict_DAS_norm['F0'][:,2])).T
    unmixed_DAS['Fm'] = dict_DAS_norm['Fm']
    
    #obtains relative amplitudes of the two PSII spectra
    dict_ampli = {}
    for key in ['F0', 'Fm']:
        y_data=unmixed_DAS[key][:,1:]

        PSII_short_sum=y_data[:,1].sum()
        PSII_long_sum=y_data[:,2].sum()
        dict_ampli[key]=(PSII_short_sum/(PSII_short_sum+PSII_long_sum),PSII_long_sum/(PSII_short_sum+PSII_long_sum))

    #calculated lifetimes
    life_times = {}
    for key in unmixed_DAS:
        PSI_lifetime = dict_time[key][0]
        PSII_lifetime = dict_time[key][1] * dict_ampli[key][0] + dict_time[key][2] * dict_ampli[key][1]
        life_times[key] = [PSI_lifetime, PSII_lifetime] 
    
    k_P2 =( 1/life_times['F0'][1] - 1/life_times['Fm'][1])*1000
    k_D2 = 1/life_times['Fm'][1] *1000
    
    #reabsorption correction
    LHCII_spectrum = get_LHCII_correction_spectrum(dict_DAS['Fm'][:,0])
    PSII_spectrum_Fm = unmixed_DAS['Fm'][:,2] + unmixed_DAS['Fm'][:,3]
    correction_factor = PSII_spectrum_Fm / LHCII_spectrum
    index_for_correction = [i for i, x in enumerate(dict_DAS['Fm'][:,0]) if ((x>750) and (x<760))]
    correction_factor_num = correction_factor[index_for_correction].mean()
    correction_factor = correction_factor / correction_factor_num
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(dict_DAS['Fm'][:,0], correction_factor)
    ax.set_xlim(650,785)
    ax.set_xlabel('Wavlength (nm)')
    ax.set_ylabel('correction factor')
    plt.savefig(os.path.join('Streak_measurements/Results',f'analysed_results\\correction_factor_{plantID}') , dpi=400)
    plt.close()

    unmixed_DAS_corrected = unmixed_DAS
    for key in unmixed_DAS:
        unmixed_DAS_corrected[key][:,1] = unmixed_DAS[key][:,1] / correction_factor
        unmixed_DAS_corrected[key][:,2] = unmixed_DAS[key][:,2] / correction_factor
        unmixed_DAS_corrected[key][:,3] = unmixed_DAS[key][:,3] / correction_factor
        
        
        
    #calculate excitation ratio
    excitation_ratios = {}
    for key in unmixed_DAS:
        ex_PSI = unmixed_DAS_corrected[key][:,1].sum()
        ex_PSII = unmixed_DAS_corrected[key][:,2].sum() + unmixed_DAS_corrected[key][:,3].sum()
        excitation_ratios[key] = ex_PSI / ex_PSII
        
        
    #plots the result
    colors = ['black', 'red', 'blue']
    titles = [r'F_o decay-associated-spectra',r'F_m decay-associated-spectra' ]
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    
    for i in range(len(unmixed_DAS.keys())):
        key = list(unmixed_DAS.keys())[i]

        ax[0,i].set_title(titles[i])
        for j in range(len(dict_DAS_norm[key].T)-1):
            ax[0,i].plot(dict_DAS[key][:,0], dict_DAS_norm[key][:,j+1]*1000, c = colors[j], label = str(round(dict_time[key][j], 1)) + ' ps')
            ax[0,i].legend()
            ax[0,i].set_xlabel('Wavelength (nm)')
            ax[0,i].set_xlim(650,785)
            ax[0,i].set_yticklabels([])
        ax[0,0].set_ylabel('Normalised intensity (a.u.)')
    
    for i in range(len(unmixed_DAS.keys())):
        key = list(unmixed_DAS.keys())[i]

        ax[1,i].set_title(titles[i])
        for j in range(len(unmixed_DAS[key].T)-1):
            ax[1,i].plot(dict_DAS[key][:,0], unmixed_DAS_corrected[key][:,j+1]*1000, c = colors[j], label = str(round(dict_time[key][j], 1)) + ' ps')
            ax[1,i].legend(loc='upper left')
            ax[1,i].set_xlabel('Wavelength (nm)')
            ax[1,i].set_xlim(650,785)
            ax[1,i].set_yticklabels([])
        ax[1,0].set_ylabel('Normalised intensity (a.u.)')
        
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    string = r'$k_{P2}$ = ' + f'{k_P2:.3f} '+ r'$ns^{-1}$' + '\n' + r'$k_{i}$ = ' + f'{k_D2:.3f} ' + r'$ns^{-1}$'
    string = string + '\nPSI/PSII excitation = ' + f'{excitation_ratios["F0"]:.3f} '
    ax[1,0].text(0.3, 0.04, string, transform=ax[1,0].transAxes, fontsize=11, bbox=props)
    string = 'PSI/PSII excitation = ' + f'{excitation_ratios["Fm"]:.3f} '
    ax[1,1].text(0.3, 0.04, string, transform=ax[1,1].transAxes, fontsize=11, bbox=props)     

    plt.savefig(os.path.join('Streak_measurements/Results',f'analysed_results\\{plantID}') , dpi=400)
    plt.close()


    fig, ax = plt.subplots(1,2, figsize=(12,5))
    

    for i in range(len(unmixed_DAS.keys())):
        key = list(unmixed_DAS.keys())[i]

        for j in range(len(unmixed_DAS[key].T)-1):
            ax[i].plot(dict_DAS[key][:,0], unmixed_DAS_corrected[key][:,j+1]*1000, c = colors[j], label = str(round(dict_time[key][j], 1)) + ' ps')
            ax[i].legend(loc='upper left')
            ax[i].set_xlabel('Wavelength (nm)')
            ax[i].set_xlim(660,785)
            ax[i].set_ylim(bottom=0 )
            ax[i].set_yticklabels([])
        ax[0].set_ylabel('Normalised intensity (a.u.)')
        
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    string = r'$k_{P2}$ = ' + f'{k_P2:.2f} '+ r'$ns^{-1}$' + '\n' + r'$k_{i}$ = ' + f'{k_D2:.2f} ' + r'$ns^{-1}$'
    ax[0].text(0.98, 0.98, string, transform=ax[0].transAxes, fontsize=11,
               bbox=props, ha='right', va='top')
    
    fig.tight_layout()
    plt.savefig(os.path.join('Streak_measurements/Results',f'analysed_results\\{plantID}_for SI') , dpi=400)
    plt.close()









    PSI_PSII_ratio_streak = excitation_ratios["Fm"]
        
    streak_data_for_NPQ = pd.DataFrame({'Photosystem':['PSII', 'PSI', 'PSII', 'PSI'],
                                        'Light condition': ['F0', 'F0', 'Fm', 'Fm'],
                                        'Lifetime':[life_times['F0'][1],life_times['F0'][0],life_times['Fm'][1],life_times['Fm'][0]],
                                        'Wavelengths': [dict_DAS['F0'][:,0],dict_DAS['F0'][:,0],dict_DAS['Fm'][:,0],dict_DAS['Fm'][:,0]],
                                        'Spectra_uncorrected':[unmixed_DAS['F0'][:,3]+unmixed_DAS['F0'][:,2],unmixed_DAS['F0'][:,1],unmixed_DAS['Fm'][:,2]+unmixed_DAS['Fm'][:,3],unmixed_DAS['Fm'][:,1]],
                                        'Spectra':[unmixed_DAS_corrected['F0'][:,3]+unmixed_DAS_corrected['F0'][:,2],unmixed_DAS_corrected['F0'][:,1],unmixed_DAS_corrected['Fm'][:,2]+unmixed_DAS_corrected['Fm'][:,3],unmixed_DAS_corrected['Fm'][:,1]]}
                                        )
    
    
    return k_P2, k_D2, PSI_PSII_ratio_streak, streak_data_for_NPQ





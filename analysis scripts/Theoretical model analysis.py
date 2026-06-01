# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:13:08 2026

@author: jans008
"""
#step function for calculating the next time step in ST analysis. 
# It takes in all the state parameters of the previous timestep, 
# individual parameters for that timestep such as Ci and light_spectrum
# and it takes general model parameters such as: Rd, Vcmax or phosphorylation rate constant.

#Finally it returns updates state variables. 
def take_time_step (Ci, light_spect,  A_prev, chloroplastic_carbon, LHCII_per_PSII, 
                    LHCII_per_PSI, D_P680, D_P700, PQH2, Fm, Fm_2, 
                    P680_excitation, P700_excitation,
                    J_P680_forward, J_P700_forward,
                    dephosphorylation_rate_constant,
                    phosphorylation_rate_constant, Vcmax, Rd, eta_LET,
                    eta_CET, gamma_star, MM_constant_oxy, MM_constant_carboxy,
                    CB6F_recovery_rate, g_meso, K_P2, K_P1, 
                    D_P680_tot, D_P700_tot, PQ_pool,
                    rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
                    open_sharing_rate, closed_sharing_rate, leaf_absorbance,
                    timestep):

    D_P680_abs = D_P680 * D_P680_tot
    D_P700_abs = D_P700 * D_P700_tot
    PQH2_abs = PQH2 * PQ_pool
    
    #calculate eta

    a = 3 + 7 * (gamma_star/chloroplastic_carbon)
    b = 4 + 8 * (gamma_star/chloroplastic_carbon) * eta_CET
    eta = 1 - (eta_LET / eta_CET) + (a / b)
    
    # update_spectra()
    PSI_LHCII_spectrum_model = D_P700_tot * LHCII_per_PSI * spectrum_LHCII
    PSI_LHCI_absorbance_spectrum = D_P700_tot * spectrum_PSI_LHCI
    absorptance_PSI = PSI_LHCII_spectrum_model + PSI_LHCI_absorbance_spectrum     

    PSII_core_spectrum_model = D_P680_tot * spectrum_CSM
    PSII_LHCII_spectrum_model = D_P680_tot * (LHCII_per_PSII-2) * spectrum_LHCII
    absorptance_PSII = PSII_core_spectrum_model + PSII_LHCII_spectrum_model 
    
    # calc_P680_excitation()
    sigma_II =  absorptance_PSII/ (absorptance_PSI + absorptance_PSII)
    P680_excitation = (light_spect * leaf_absorbance * sigma_II).sum(axis=0)

    # calc_P700_excitation()
    sigma_I = absorptance_PSI/ (absorptance_PSI +  absorptance_PSII)
    P700_excitation = (light_spect * leaf_absorbance * sigma_I).sum(axis=0)
    

    # analytic solution is determined 5 times in a row to set the phi_U2
    # to the equilibrium value. This is because phi_U2 is dependent on
    # D_680. Empiric analysis shows that 5 repeats is enough to 
    # find the equilibrium value. 
    for _ in range(5):
        # phi_U2 calculation
        phi_U2 = (D_P680_abs / D_P680_tot) * open_sharing_rate + (1 - (D_P680_abs / D_P680_tot)) * closed_sharing_rate
        # PQH2 calculation
        Phi_P680_photo = K_P2 / rate_tot_P680_open
        a_val = PQ_pool * Phi_P680_photo * P680_excitation * eta  
        b_val = (CB6F_recovery_rate * (1 - phi_U2)) + (Phi_P680_photo * P680_excitation * eta)
        PQH2_abs = a_val / b_val
        
    
        # D_P700_open calculation
        Phi_P700_photo = K_P1 / rate_tot_P700_open
        D_P700_abs = (PQH2_abs * CB6F_recovery_rate * D_P700_tot) / (PQ_pool * Phi_P700_photo * P700_excitation)
        # D_P680_open calculation
        D_P680_abs = (1 - (PQH2_abs / PQ_pool)) * D_P680_tot
        
        if D_P700_abs > D_P700_tot:
            D_P680_abs = ((Phi_P700_photo * P700_excitation) / eta) * ((1 - phi_U2) / (Phi_P680_photo * P680_excitation)) * D_P680_tot
    
            PQH2_abs =  (1 - (D_P680_abs / D_P680_tot)) * PQ_pool
            D_P700_abs = D_P700_tot
    
    #calc_J_P680_forward()
    Phi_P680_photo = K_P2 / rate_tot_P680_open
    J_P680_forward = D_P680_abs * Phi_P680_photo * P680_excitation / (D_P680_tot * (1 - phi_U2))

    #calc_J_P700_forward()
    Phi_P700_photo = K_P1 / rate_tot_P700_open
    J_P700_forward = D_P700_abs * Phi_P700_photo * P700_excitation / D_P700_tot 

    for _ in range(5):
        W_J = (chloroplastic_carbon * J_P680_forward) / (4 * chloroplastic_carbon + 8 * gamma_star)
        W_c = (chloroplastic_carbon * Vcmax) / (chloroplastic_carbon + MM_constant_carboxy * (1 + (chloroplastic_oxygen / MM_constant_oxy)))
        A = (1 - (gamma_star / chloroplastic_carbon)) * min(W_c, W_J) - Rd
        chloroplastic_carbon = Ci - (A / g_meso)

    
    #updating LHCII distribution
    LHCII_dephosphorylated = LHCII_per_PSII * D_P680_tot 
    LHCII_phosphorylated = LHCII_per_PSI * D_P700_tot 
    dephosphorylation_rate = LHCII_phosphorylated * dephosphorylation_rate_constant
    phosphorylation_rate = LHCII_dephosphorylated * phosphorylation_rate_constant * (PQH2_abs / PQ_pool)
    
    change_rate = (phosphorylation_rate - dephosphorylation_rate) * timestep
    LHCII_phosphorylated += change_rate 
    LHCII_dephosphorylated -= change_rate 
    
    #if phosphorylation is outside of the possible bounds, it is set to the
    #most extreme possible distribution within the bounds.
    #add these back if needed
    #LHCII_dephosphorylated = pt.minimum(LHCII_dephosphorylated + LHCII_phosphorylated, LHCII_dephosphorylated)
    #LHCII_phosphorylated = pt.maximum(0, LHCII_phosphorylated)

    #LHCII_phosphorylated = pt.minimum(LHCII_phosphorylated + LHCII_dephosphorylated, LHCII_phosphorylated)
    #LHCII_dephosphorylated = pt.maximum(0, LHCII_dephosphorylated)
    
        
    LHCII_per_PSII_new = LHCII_dephosphorylated / D_P680_tot
    LHCII_per_PSI_new = LHCII_phosphorylated / D_P700_tot
    
    
    #calculations for qT
    excitation_PSII_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_II).sum(axis=0)
    excitation_PSI_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_I).sum(axis=0)
    
    a = excitation_PSII_PAM * epsilon_F2 * (K_F2 / (rate_tot_P680_closed-K_U2))
    b = excitation_PSI_PAM * epsilon_F1 * (K_F1 / rate_tot_P700_closed)
    Fm = a + b
    Fm_2 = a
    
    D_P680 = D_P680_abs / D_P680_tot
    D_P700 = D_P700_abs / D_P700_tot
    PQH2 = PQH2_abs / PQ_pool
    
    new_state = (A, chloroplastic_carbon, LHCII_per_PSII_new, LHCII_per_PSI_new, D_P680, D_P700, PQH2, Fm, Fm_2, P680_excitation, P700_excitation, J_P680_forward, J_P700_forward)

    return np.array(new_state)


if __name__ == "__main__":
    import arviz as az
    
    import matplotlib.pyplot as plt
    
    import numpy as np
    import pandas as pd
    import pymc as pm
    import os
    import pickle as pkl
    import pytensor
    import pytensor.tensor as pt
    import pymc_extras
    import seaborn as sns
    import pickle as pkl
    import math
    from initial_data_processing import plant_instance
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    import statsmodels.api as sm
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    


    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    base_directory = os.path.dirname(os.path.realpath(__file__))
    
    output_folder = 'theoretical figures/'
    #define parameters

    gamma_star = 40*1e-6 # bar at 25C, standard value from Duursma 2015 and Bernacchi 2001  
    MM_constant_oxy = 175*1e-3 # bar at 25C,  Duursma 2015 and Bernacchi 2001
    MM_constant_carboxy = 300*1e-6 # bar at 25C, Duursma 2015 and Bernacchi 2001
    chloroplastic_oxygen = 212.7*1e-3 # bar, 21% of atmospheric pressure
    g_meso = 0.3 #mol m-2 s-1 bar-1 
    Vcmax = 100*1e-6# random find source
    Rd_standard = 1*1e-6 #random find source

    Ci = 280*1e-6
    
    #set rate parameters in PSI and PSII
  
    K_P2 = 3.3 # ns-1, rate of photochemistry
    K_N2_standard = 0 # ns-1, assumed at 0 under light limiting conditions
    K_D2 = 0.55 # ns-1, rate of constitutive heat loss
    K_F2 = 0.05 # ns-1, rate of fluorescence
    global K_U2
    K_U2 = 2.1 # ns-1, to do
    
    K_P1 = 16 # ns-1, rate of photochemistry
    K_D1 = 0.55 # ns-1, rate of constitutive heat loss
    K_F1 = 0.05 # ns-1, rate of fluorescence
    K_X1 = K_P1 # ns-1, rate of dissipation in closed P700, set equal to K_P1
    rate_tot_P700_open = K_P1 + K_D1 + K_F1
    rate_tot_P700_closed = K_X1 + K_D1 + K_F1

    epsilon_F1 = 2
    epsilon_F2 = 1
    
    # Set initial number of thylakoid membrane components.

    Chl_per_area_standard = 0.00042 # mol m-2 
    PS_per_chl = 0.0046 # mol/mol chl, Ibrahim 2022,
    PSI_PSII_ratio = 0.6 #PSI/PSII, McKenzie 2020

    concentration_CB6F_standard = 0.00076 # mol/mol Chl, ibrahim 2022
    PQ_per_chl = 0.01 # mol/mol chl, kruk 2006
    LHCII_per_PSII = 3
    LHCII_per_PSI =1
    chl_ab = 2.9 #random value needs source
    
    # State number of chlorophylls per protein complex.
    PS_table = pd.DataFrame({'chl_a': [148, 108, 24],
                             'chl_b': [12, 49, 18],
                             'chl_tot': [160, 157, 42]}, index=['PSI-LHCI', 'CSM', 'LHCII'])
    #PSI-LHCI from Pan 2018; CSM from Su 2014; LHCII from Laisk 2014
    
    k_CB6F_star = 300 # mol PQH2 mol-1 sites s-1, catalytic constant for Cyt B6F assuming relaxed photosynthetic control
    
    eta_LET = 0.75 # mol ATP mol-1 e-1
    eta_CET = 1 # mol ATP mol-1 e-1
    
        
    #import leaf absorbance and absorbance by photosynthetic pigments. 
    parameters = pd.read_excel('model parameters fvCBwd model/parameters.xlsx', header=2)
    # Inter- and extrapolates the missing values in the absorption by non-photosynthetic pigmens.  
    known_abs_pig = np.transpose(np.array(parameters[parameters['alpha_pigments'].notna()][['Wavelength(nm)', 'alpha_pigments']]))
    x_new = np.array(parameters['Wavelength(nm)'])
    parameters['alpha_pigments'] = np.interp(x_new, known_abs_pig[0], known_abs_pig[1])
    absorbance_photopigments = np.array(parameters.alpha_pigments)
    leaf_absorbance_lit = np.array(parameters.leaf_absorbance)
        
    test_plant = 'NT9'
    with open(f'pickles_with_priors/{test_plant}.pkl', 'rb') as file:
        plant_test = pkl.load(file)
        leaf_absorbance_NT = plant_test.leaf_absorbance
    
        
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
    
    spectrum_480 = load_spectrum('480_light', 40) #Correction factor is applied to get 40 umol absorbed photons
    spectrum_700 = load_spectrum('700_light', 55) #Correction factor is applied to get 40 umol absorbed photons        

    
    #measured spectrum of the Dual-PAM fluorescence measuring light 
    PAM_spect = np.loadtxt('Dual_pam_measuring_light_spectrum.txt', delimiter='\t', skiprows=1)[:371,1]
    PAM_spect[:50] = 0
    
    #import absorbance spectra of protein components.
    file_names = ['Hogewoning_PSI', 'Hogewoning_PSII_core', 'Hogewoning_LHCII']
    spectra = np.zeros((3,371))
    for i in range(len(file_names)):
        file_name = 'model parameters FvCBwd model/' + file_names[i] + '.csv'
        file = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
        spectrum = np.array(file.intensity)
        spectra[i] = spectrum
        
    PSI_LHCI_chl = PS_table.at['PSI-LHCI', 'chl_a'] + 0.7 * PS_table.at['PSI-LHCI', 'chl_b']
    spectrum_PSI_LHCI = (PSI_LHCI_chl / spectra[0][250:].sum(axis=0)) * (spectra[0])
    
    CSM_chl = PS_table.at['CSM', 'chl_a'] + 0.7 * PS_table.at['CSM', 'chl_b']
    spectrum_CSM = (CSM_chl / spectra[1][250:].sum(axis=0)) * (spectra[1])
    
    LHCII_chl = PS_table.at['LHCII', 'chl_a'] + 0.7 * PS_table.at['LHCII', 'chl_b']
    spectrum_LHCII = (LHCII_chl / spectra[2][250:].sum(axis=0)) * (spectra[2])
     

    

    
    PSI_PSII_ratio_standard = 0.60
    
    # sets state transition rate constants
    dephosphorylation_rate_constant_standard = 0.5/60 #0.005
    phosphorylation_rate_constant_standard = 0.2/60
    
    
    #initialise light and simulation time
    section_minutes = 15
    section_time = section_minutes * 60
    time_steps_model_standard = 5
    model_length_standard =int( 3 * section_time / time_steps_model_standard)
    Ci_model = np.repeat([Ci], model_length_standard)
    Time_standard = np.arange(-1 * section_time/60,2 * section_time/60, time_steps_model_standard/60)

    t_plot = (0, 2*section_time/60)
    
    
    
 
    fig, ax = plt.subplots()

    ax.plot(np.arange(380,751, 1), spectrum_480*1e6, label='480 lamp')
    ax.plot(np.arange(380,751, 1), spectrum_700*1e6, label='700 lamp')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('wavelength(nm)')
    ax.set_ylabel('light intensity umol m-2 s-1 nm-1')  
    ax.set_xlim(380, 730)
    ax.legend(ncols=1, loc='lower center', bbox_to_anchor=(0.5,1))
    #ax.set_ylim(-0.1, 1)

    plt.show()
    
    
    light_list_standard = np.zeros((model_length_standard,371))
    for i in range(len(light_list_standard)):
        if (i>=len(light_list_standard)/3) and (i<len(light_list_standard)*2/3):
            light_list_standard[i] = spectrum_480
        else:
            light_list_standard[i] = spectrum_700
    
    

    
def run_model(PSI_PSII_ratio=PSI_PSII_ratio_standard, 
              dephosphorylation_rate_constant=dephosphorylation_rate_constant_standard, 
              phosphorylation_rate_constant=phosphorylation_rate_constant_standard,
              K_N2 = K_N2_standard,
              Chl_per_area=Chl_per_area_standard,
              concentration_CB6F = concentration_CB6F_standard,
              light_list = light_list_standard,
              model_length=model_length_standard,
              Time = Time_standard,
              cutoff = 3,
              Rd = Rd_standard,
              time_steps_model = time_steps_model_standard,
              leaf_absorbance = leaf_absorbance_lit):
    #initialise thylakoid membrane
    concentration_PSII = PS_per_chl * (1 - (PSI_PSII_ratio / (1 + PSI_PSII_ratio))) # mol/mol Chl
    concentration_PSI = PS_per_chl - concentration_PSII # mol/mol Chl
    D_P680_tot = concentration_PSII * Chl_per_area # mol m-2
    D_P700_tot = concentration_PSI * Chl_per_area # mol m-2
    D_CB6F_tot = concentration_CB6F * Chl_per_area # mol m-2
    CB6F_recovery_rate = D_CB6F_tot * k_CB6F_star #  mol PQH2 m-2 s-1
    PQ_pool = PQ_per_chl * Chl_per_area # mol m-2
    
    #calculate LHCII amount from PSI/PSII ratio and from chl_ab ratio
    a = (PS_table.at['PSI-LHCI', 'chl_a'] - chl_ab * PS_table.at['PSI-LHCI', 'chl_b']) / (1 + (1/PSI_PSII_ratio))
    b = (PS_table.at['CSM', 'chl_a'] - chl_ab * PS_table.at['CSM', 'chl_b'])/ (1 + PSI_PSII_ratio)
    c = chl_ab * PS_table.at['LHCII', 'chl_b'] - PS_table.at['LHCII', 'chl_a']
    extra_LHCII_per_PS = (a + b) / c
    
        

    
   
    #defining ST model parameters
    rate_tot_P680_open = K_P2 + K_N2 + K_D2 + K_F2 + K_U2
    rate_tot_P680_closed = K_N2 + K_D2 + K_F2 + K_U2
    open_sharing_rate = K_U2 / rate_tot_P680_open
    closed_sharing_rate = K_U2 / rate_tot_P680_closed

    
    #initalise initial state
    A = 0
    chloroplastic_carbon_init = Ci
    LHCII_per_PSII_init = 2 + extra_LHCII_per_PS #LHCII_per_PSII_init
    LHCII_per_PSI_init = extra_LHCII_per_PS #LHCII_per_PSI_init
    D_P680_init = 1 #D_P680 is initiated fully open
    D_P700_init = 1 #D_P680 is initiated fully open
    PQH2_init = 0
    Fm_init = 1 # placeholder for Fm, value is not used
    Fm_2_init = 1 # placeholder for Fm, value is not used
    P680_excitation_init = 40*1e-6 # placeholder for excitation, value is not used
    P700_excitation_init = 40*1e-6 # placeholder for excitation, value is not used
    J_P680_forward_init = 40*1e-6 # placeholder for flow rate, value is not used
    J_P700_forward_init = 40*1e-6 # placeholder for flow rate, value is not used
    
    
    
    results = np.zeros((model_length+1,13))
    results[0] = [A, chloroplastic_carbon_init, LHCII_per_PSII_init, LHCII_per_PSI_init,
                  D_P680_init, D_P700_init, PQH2_init, Fm_init, Fm_2_init, P680_excitation_init,
                  P700_excitation_init, J_P680_forward_init, J_P700_forward_init]
          
    for i in range(model_length):
        results[i+1,:] = take_time_step(Ci, light_list[i], results[i,0], results[i,1], results[i,2],
                                      results[i,3], results[i,4], results[i,5], results[i,6], results[i,7], results[i,8],
                                      results[i,9], results[i,10],
                                      results[i,11], results[i,12],
                                      dephosphorylation_rate_constant,
                                      phosphorylation_rate_constant, Vcmax, Rd, eta_LET,
                                      eta_CET, gamma_star, MM_constant_oxy, MM_constant_carboxy,
                                      CB6F_recovery_rate, g_meso, K_P2, K_P1, 
                                      D_P680_tot, D_P700_tot, PQ_pool,
                                      rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
                                      open_sharing_rate, closed_sharing_rate, leaf_absorbance,
                                      time_steps_model)
        
        
    #Convert to useful units
    results[:,0] = results[:,0]*1e6 #A
    results[:,1] = results[:,1]*1e6 #Cc
    results[:,9] = results[:,9]*1e6 #P580 excitation
    results[:,10] = results[:,10]*1e6 #P700 excitation
    results[:,11] = results[:,11]*1e6 #J_P680 forward
    results[:,12] = results[:,12]*1e6 #J_P700 forward

    
    index_Fm = int(model_length*1/3)+1
    qT = 100*((results[:,7][index_Fm] - results[:,7]) / results[:,7][index_Fm])
    
    results_frame = pd.DataFrame(results, 
                                 columns=['A', 'Cc', 'LHCII_per_PSII',
                                          'LHCII_per_PSI', 'D_P680', 'D_P700',
                                          'PQH2', 'Fm', 'Fm2', 'P680_excitation',
                                          'P700_excitation', 'J_P680_forward', 'J_P700_forward'])
    results_frame['qT'] = qT
    results_frame['Time'] = np.concatenate((np.array([0]), Time), axis=None)
    results_frame.drop([0,], inplace=True)

    results_frame = results_frame.iloc[int(len(results_frame)/cutoff):]
    return results_frame

results_frame_standard = run_model(leaf_absorbance=leaf_absorbance_lit)
#%%Plot results
fontsize=12
def make_theoretical_plot(results_frame, save_name):
    fig, axs = plt.subplots(2,3, figsize=(12,8))
    
    axs[0,0].plot(results_frame.Time, results_frame.P680_excitation, ls='-', color='black', label="PSII excitation")
    axs[0,0].plot(results_frame.Time, results_frame.P700_excitation, ls='--', color='black', label="PSI excitation")
    axs[0,0].set_ylabel(r'Photosystem excitation $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[0,0].set_ylim(0, results_frame.P700_excitation.max()+5)
    axs[0,0].legend(fontsize=fontsize)
    
    
    P680_plot, = axs[0,1].plot(results_frame.Time, results_frame.D_P680, ls='-', color='black', label="P680")
    P700_plot, = axs[0,1].plot(results_frame.Time, results_frame.D_P700, ls='--', color='black', label="P700")
    PQ_ax = axs[0,1].twinx()
    PQ_plot, = PQ_ax.plot(results_frame.Time, results_frame.PQH2, color='orange', ls='-', label=r"$PQH_2$/$PQ_{tot}$")
    axs[0,1].set_ylabel('Fraction open reaction centres', fontsize=fontsize)
    PQ_ax.set_ylabel(r'$PQH_2/PQ_{tot}$', fontsize=fontsize)
    axs[0,1].set_ylim(-0.22, 1.1)
    PQ_ax.set_ylim(-0.22, 1.1)
    axs[0,1].legend(handles=[P680_plot, P700_plot, PQ_plot], loc='lower right', fontsize=fontsize)
    
        
    QY = results_frame.D_P700
    axs[0,2].plot(results_frame.Time,1-QY , ls='--',c='orange', label='P700 donor limitation')
    axs[0,2].plot(results_frame.Time, QY, ls='--', c='black', label='P700 photochemistry')
    axs[0,2].set_ylabel('QY', fontsize=fontsize)
    axs[0,2].legend(loc='lower right', fontsize=fontsize)
    axs[0,2].set_ylim(-0.25,1.1)
    
    axs[1,0].plot(results_frame.Time, results_frame.LHCII_per_PSII, ls='-', color='black', label="LHCII per PSII")
    axs[1,0].plot(results_frame.Time, results_frame.LHCII_per_PSI, ls='--', color='black', label="LHCII per PSI")
    axs[1,0].set_ylabel('LHCII per PS', fontsize=fontsize)
    axs[1,0].set_ylim(0,results_frame.LHCII_per_PSII.max()+0.4)
    axs[1,0].legend(loc='lower left', fontsize=fontsize)
    
    
    axs[1,1].plot(results_frame.Time, results_frame.J_P680_forward, ls='-', color='black', label="PSII electron flow ")
    axs[1,1].plot(results_frame.Time, results_frame.J_P700_forward, ls='--', color='black', label="PSI electron flow")
    axs[1,1].set_ylabel(r'Electron flow rate $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[1,1].set_ylim(0,results_frame.J_P700_forward.max()+5)
    axs[1,1].legend(loc='lower right', fontsize=fontsize)
    
    A_plot, = axs[1,2].plot(results_frame.Time, results_frame.A, color='orange', label='Net Assimilation')
    axs[1,2].set_ylabel(r'Net Assimilation  $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
    axs[1,2].set_ylim(0,max(results_frame.A)+0.4)
    qT_ax = axs[1,2].twinx()
    qT_plot, = qT_ax.plot(results_frame.Time, results_frame.qT, color='blue', ls='-', label='qT')
    qT_ax.set_ylabel('qT (%)', fontsize=fontsize)
    qT_ax.set_ylim(0,12)
    axs[1,2].legend(handles = [A_plot, qT_plot], loc='lower left', fontsize=fontsize)
    

    
    
    colors = ['lightskyblue', 'lightcoral']
    texts = ['Blue LED', 'FR LED']
    for i in range(2):
        for j in range(3):
            axs[i,j].tick_params(axis='both', which='major', labelsize=fontsize)
            axs[i,j].set_xlim(t_plot[0], t_plot[1])
            axs[1,j].set_xlabel('Time (min)')
            for k in range(2):
                
                banner = mpatches.Rectangle((0+k*0.5, 0.94), 0.5+k*0.5, 0.06,
                                            transform=axs[i,j].transAxes,  # axes coords
                                            facecolor=colors[k],
                                            edgecolor=colors[k], linewidth=0.8)
                axs[i,j].add_patch(banner)
                axs[i,j].text(0.25+k*0.5, 0.975, texts[k], transform=axs[i,j].transAxes,
                        va="center_baseline", ha="center", fontsize=12, color="black")
                axs[i,j].vlines(section_minutes,-10,150, ls='--', color='lightgrey', zorder=0)
    axs[1,1].set_yticks([0,5,10,15,20])
    axs[1,2].set_yticks([0,0.5,1,1.5,2])
    
    fig.text(0.01,0.97, 'A', fontsize=15)
    fig.text(0.33,0.97, 'B', fontsize=15)
    fig.text(0.66,0.97, 'C', fontsize=15)
    
    fig.text(0.01,0.49, 'D', fontsize=15)
    fig.text(0.33,0.49, 'E', fontsize=15)
    fig.text(0.66,0.49, 'F', fontsize=15)
    
    
    fig.tight_layout()
    save_name = os.path.join(output_folder,save_name+ '.png')
    fig.savefig(save_name,dpi=400)
    plt.show()
    plt.close() 
    
    
make_theoretical_plot(results_frame_standard, 'theoretical_figure')

#%%Case studies
fontsize=11
PS_ratio_list = [0.6*PSI_PSII_ratio_standard, 0.8*PSI_PSII_ratio_standard,PSI_PSII_ratio_standard, 
                 1.2*PSI_PSII_ratio_standard, 1.4*PSI_PSII_ratio_standard]
results_frame_list_PS = []

for i in range(len(PS_ratio_list)):
    PSI_PSII_ratio = PS_ratio_list[i]
    results_frame_list_PS.append(run_model(PSI_PSII_ratio=PSI_PSII_ratio, leaf_absorbance=leaf_absorbance_lit))

NPQ_list = np.array([0,1,2,3,4])
NPQ_rate_list = NPQ_list * (K_D2 + K_F2)

results_frame_list_NPQ = []

for i in range(len(NPQ_rate_list)):
    NPQ_rate = NPQ_rate_list[i]
    results_frame_list_NPQ.append(run_model(K_N2=NPQ_rate, leaf_absorbance=leaf_absorbance_lit))
    

rates_list_abs = [(dephosphorylation_rate_constant_standard*0.6, phosphorylation_rate_constant_standard*0.6),
              (dephosphorylation_rate_constant_standard*0.8, phosphorylation_rate_constant_standard*0.8),
              (dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard),
              (dephosphorylation_rate_constant_standard*1.5, phosphorylation_rate_constant_standard*1.5),
              (dephosphorylation_rate_constant_standard*2, phosphorylation_rate_constant_standard*2),]
results_frame_abs_list = []

for i in range(len(rates_list_abs)):
    dephos_rate, phos_rate = rates_list_abs[i]

    results_frame_abs_list.append(run_model(dephosphorylation_rate_constant=dephos_rate,
                                        phosphorylation_rate_constant=phos_rate,
                                        leaf_absorbance=leaf_absorbance_lit))
                     

rates_list_rel = [(dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard*0.6),
              (dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard*0.8),
              (dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard),
              (dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard*1.5),
              (dephosphorylation_rate_constant_standard, phosphorylation_rate_constant_standard*2),]
results_frame_rel_list = []

for i in range(len(rates_list_rel)):
    dephos_rate, phos_rate = rates_list_rel[i]

    results_frame_rel_list.append(run_model(dephosphorylation_rate_constant=dephos_rate,
                                        phosphorylation_rate_constant=phos_rate,
                                        leaf_absorbance=leaf_absorbance_lit))
        
data = [results_frame_rel_list, results_frame_abs_list, results_frame_list_PS, results_frame_list_NPQ]
fig, axs = plt.subplots(4,3, figsize=(12,16))
colours = ['black', 'dimgrey', 'darkgrey', 'silver', 'lightgrey']
for i in range(4):
    results_frame_list = data[i]
    for j in range(len(results_frame_list)):
        results_frame = results_frame_list[j]
        
        axs[i,0].plot(results_frame.Time, results_frame.LHCII_per_PSII, ls='-', color=colours[j], label="LHCII per PSII")
        axs[i,0].plot(results_frame.Time, results_frame.LHCII_per_PSI, ls='--', color=colours[j], label="LHCII per PSI")
        axs[i,0].set_ylabel('LHCII per PS', fontsize=fontsize)
        axs[i,0].set_ylim(0,5)
        
        
        A_plot, = axs[i,1].plot(results_frame.Time, results_frame.A, color=colours[j], label='Net Assimilation')
        axs[i,1].set_ylabel(r'Net Assimilation  $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
        axs[i,1].set_ylim(0,2.2)

        
        qT_plot, = axs[i,2].plot(results_frame.Time, results_frame.qT, color=colours[j], ls='-', label='qT')
        axs[i,2].set_ylabel('qT (%)', fontsize=fontsize)
        axs[i,2].set_ylim(0,15)
    
    
colors = ['lightskyblue', 'lightcoral']
texts = ['Blue LED', 'FR LED']
for i in range(4):
    for j in range(3):
        axs[i,j].set_xlim(t_plot[0], t_plot[1])
        axs[3,j].set_xlabel('Time (min)', fontsize=fontsize)
        axs[i,1].set_yticks([0,0.5,1,1.5,2])
        for k in range(2):
            
            banner = mpatches.Rectangle((0+k*0.5, 0.94), 0.5+k*0.5, 0.06,
                                        transform=axs[i,j].transAxes,  # axes coords
                                        facecolor=colors[k],
                                        edgecolor=colors[k], linewidth=0.8)
            axs[i,j].add_patch(banner)
            axs[i,j].text(0.25+k*0.5, 0.975, texts[k], transform=axs[i,j].transAxes,
                    va="center_baseline", ha="center", fontsize=12, color="black")
            axs[i,j].vlines(section_minutes,-10,150, ls='--', color='lightgrey', zorder=0)


PS_lines = [Line2D([0], [0], color='black', linestyle='-'),
            Line2D([0], [0], color='black', linestyle='--')]
PS_strings = ['PSII', 'PSI']

legend_lines = []
legend_strings = []
for i in range(len(rates_list_rel)):
    legend_lines.append(Line2D([0], [0], color=colours[i], linestyle='-'))
    legend_strings.append(r'$k_{1\,\to\,2}$/$k_{2\,\to\,1}$ = '+ f'{rates_list_rel[i][1]/rates_list_rel[i][0]:.2f}')

axs[0,2].legend(legend_lines,legend_strings, fontsize=fontsize, loc='upper right', 
                title= r'$k_{2\,\to\,1}$ = ' +f'{rates_list_rel[0][0]*60:.2f}' + r' $min^{-1}$',
                bbox_to_anchor = [1.01,0.95])
axs[0,0].legend(PS_lines, PS_strings, fontsize=fontsize, loc='upper right',
                bbox_to_anchor = [1.01,0.95])


legend_lines = []
legend_strings = []
for i in range(len(rates_list_abs)):
    legend_lines.append(Line2D([0], [0], color=colours[i], linestyle='-'))
    legend_strings.append(r'$k_{2\,\to\,1}$ = '+ f'{rates_list_abs[i][0]*60:.2f}'+ r' $min^{-1}$')
axs[1,2].legend(legend_lines,legend_strings, fontsize=fontsize, loc='upper right', 
                title= r'$k_{1\,\to\,2}$/$k_{2\,\to\,1}$ = ' +f'{rates_list_abs[2][1]/rates_list_abs[2][0]:.2f}',
                bbox_to_anchor = [1.01,0.95])

axs[1,0].legend(PS_lines, PS_strings, fontsize=fontsize, loc='upper right',
                bbox_to_anchor = [1.01,0.95])


legend_lines = []
legend_strings = []
for i in range(len(PS_ratio_list)):
    legend_lines.append(Line2D([0], [0], color=colours[i], linestyle='-'))
    legend_strings.append(r'PSI/PSII = '+ f'{PS_ratio_list[i]:.2f}')
axs[2,2].legend(legend_lines,legend_strings, fontsize=fontsize, loc='upper right',
                bbox_to_anchor = [1.01,0.95])
PS_lines = [Line2D([0], [0], color='black', linestyle='-'),
            Line2D([0], [0], color='black', linestyle='--')]
PS_strings = ['PSII', 'PSI']
axs[2,0].legend(PS_lines, PS_strings, fontsize=fontsize, loc='upper right',
                bbox_to_anchor = [1.01,0.95])

legend_lines = []
legend_strings = []

for i in range(len(NPQ_list)):
    legend_lines.append(Line2D([0], [0], color=colours[i], linestyle='-'))
    legend_strings.append('NPQ = '+ f'{NPQ_list[i]:.2f}')

axs[3,2].legend(legend_lines,legend_strings, fontsize=fontsize, 
                loc='upper right',bbox_to_anchor = [1.01,0.95])
axs[3,0].legend(PS_lines, PS_strings, fontsize=fontsize, loc='upper right',
                bbox_to_anchor = [1.01,0.95])


fig.text(0.01,0.98, 'A', fontsize=15)
fig.text(0.35,0.98, 'B', fontsize=15)
fig.text(0.67,0.98, 'C', fontsize=15)

fig.text(0.01,0.73, 'D', fontsize=15)
fig.text(0.35,0.73, 'E', fontsize=15)
fig.text(0.67,0.73, 'F', fontsize=15)

fig.text(0.01,0.49, 'G', fontsize=15)
fig.text(0.35,0.49, 'H', fontsize=15)
fig.text(0.67,0.49, 'I', fontsize=15)

fig.text(0.01,0.24, 'J', fontsize=15)
fig.text(0.35,0.24, 'K', fontsize=15)
fig.text(0.67,0.24, 'L', fontsize=15)


fig.tight_layout()
save_name = os.path.join(output_folder,'scenarios_plot.png')
fig.savefig(save_name,dpi=400)
plt.show()
plt.close() 


#%% Mattila data
def find_opt_PS_ratio(growth_light_spectrum, phos_rate = phosphorylation_rate_constant_standard,
                      dephos_rate=dephosphorylation_rate_constant_standard,
                      plot_figures=False):
    test_length = 1000
    
    
    test_light = np.zeros((test_length,371))
    for i in range(test_length):
        test_light[i] = growth_light_spectrum

    test_time = np.arange(0,5*test_length, 5)
    
    
    PS_test = np.arange(0.1,2,0.1)
    A_test = []
    for i in range(len(PS_test)):
        results_frame = run_model(PSI_PSII_ratio=PS_test[i], 
                                  dephosphorylation_rate_constant=dephos_rate, 
                                  phosphorylation_rate_constant=phos_rate,
                                  K_N2 = 0,
                                  light_list = test_light,
                                  model_length=test_length,
                                  Time = test_time,
                                  leaf_absorbance=leaf_absorbance_lit)
        A_test.append(results_frame.A.iat[-1])
    A_test = np.array(A_test)
    max_A = max(A_test)
    opt_PS = PS_test[np.where(A_test ==  max_A)][0]
    PS_test_2 = np.arange(opt_PS-0.1,opt_PS+0.11,0.01)

    A_test_2 = []
    for i in range(len(PS_test_2)):
        results_frame = run_model(PSI_PSII_ratio=PS_test_2[i], 
                                  dephosphorylation_rate_constant=dephos_rate, 
                                  phosphorylation_rate_constant=phos_rate,
                                  K_N2 = 0,
                                  light_list = test_light,
                                  model_length=test_length,
                                  Time = test_time, 
                                  leaf_absorbance=leaf_absorbance_lit)
        A_test_2.append(results_frame.A.iat[-1])
    A_test_2 = np.array(A_test_2)
    max_A_2 = max(A_test_2)
    opt_PS_2 = PS_test_2[np.where(A_test_2 ==  max_A_2)][0]
    
    if plot_figures:    
        fig, ax = plt.subplots(1,1)
        ax.plot(PS_test, A_test)
        ax.set_title('CB6F per chl:'+ str(concentration_CB6F_standard) + 'chl content:'+ str(Chl_per_area_standard))
        ax.set_xlabel('PS_ratio')
        ax.set_ylabel('Net_assimilation under growth light')
        plt.show()
    return A_test_2, opt_PS_2

#load spectrum from csv file and calculate spectrum list from 380 to 730 nm.    
file_name = 'mattila data/growth_light.csv'
spectrum_df = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
spectrum_mattila_growth = np.zeros(371)
if spectrum_df.wavelength[0] < 380:
    print('Watch out, spectrum provided have wavelengths lower than 380, this will lead to errors')
else: 
    offset = int(spectrum_df.wavelength[0]-380)
    max_index = min(len(spectrum_df.wavelength), 370-offset)
    spectrum_mattila_growth[offset:offset+max_index] = spectrum_df.intensity[0:max_index]
    
    #normalises spectrum to light intensity
    light_intensity = 150
    tot_intensity = spectrum_mattila_growth.sum()
    correction_factor = light_intensity / tot_intensity
    spectrum_mattila_growth = spectrum_mattila_growth * correction_factor * 1e-6
    

    
    
assimilation, opt_PS_ratio_mattila = find_opt_PS_ratio(spectrum_mattila_growth, plot_figures=True)  


#%% Testing CB6F and opt PS ratio
def find_opt_PS_ratio_CB6F(growth_light_spectrum, Chl_per_area=Chl_per_area_standard, 
                           phos_rate = phosphorylation_rate_constant_standard,
                      dephos_rate=dephosphorylation_rate_constant_standard,
                      concentration_CB6F = concentration_CB6F_standard,
                      plot_figures=False):
    test_length = 1000
    
    
    test_light = np.zeros((test_length,371))
    for i in range(test_length):
        test_light[i] = growth_light_spectrum

    test_time = np.arange(0,5*test_length, 5)
    
    
    PS_test = np.arange(0.1,2,0.1)
    A_test = []
    for i in range(len(PS_test)):
        results_frame = run_model(PSI_PSII_ratio=PS_test[i], 
                                  dephosphorylation_rate_constant=dephos_rate, 
                                  phosphorylation_rate_constant=phos_rate,
                                  Chl_per_area = Chl_per_area,
                                  concentration_CB6F=concentration_CB6F,
                                  K_N2 = 0,
                                  light_list = test_light,
                                  model_length=test_length,
                                  Time = test_time,
                                  leaf_absorbance=leaf_absorbance_lit)
        A_test.append(results_frame.A.iat[-1])
    A_test = np.array(A_test)
    max_A = max(A_test)
    opt_PS = PS_test[np.where(A_test ==  max_A)][0]

    if plot_figures:    
        fig, ax = plt.subplots(1,1)
        ax.plot(PS_test, A_test)
        ax.set_title('CB6F per chl:'+ str(concentration_CB6F_standard) + 'chl content:'+ str(Chl_per_area_standard))
        ax.set_xlabel('PS_ratio')
        ax.set_ylabel('Net_assimilation under growth light')
        plt.show()
    return A_test, opt_PS


chl_contents = np.arange(0.0002, 0.0011, 0.0001)
fig, ax = plt.subplots(1,1)
PS_test = np.arange(0.1,2,0.1)
for chl_content in chl_contents:
    assimilation, opt_PS_ratio = find_opt_PS_ratio_CB6F(spectrum_mattila_growth, Chl_per_area = chl_content, plot_figures=False)  
    ax.plot(PS_test, assimilation,label=str(round(chl_content*1e6,4))+' umol m-2')
ax.set_xlabel('PS_ratio')
ax.set_ylabel('Net_assimilation under growth light')
ax.legend(title = 'chl content')
plt.show()

CB6F_contents = np.arange(0.0001,0.002,0.0001)
fig, ax = plt.subplots(1,1)
PS_test = np.arange(0.1,2,0.1)
for CB6F_content in CB6F_contents:
    assimilation, opt_PS_ratio = find_opt_PS_ratio_CB6F(spectrum_mattila_growth, concentration_CB6F=CB6F_content, plot_figures=False)  
    ax.plot(PS_test, assimilation,label=str(round(CB6F_content,4))+' umol m-2')
ax.set_xlabel('PS_ratio')
ax.set_ylabel('Net_assimilation under growth light')
ax.legend(title = 'concentration_CB6F')
plt.show()
#%%
def run_mattila_lamp_test(lamp_name, light_intensity, NPQ_rate=0):
    #load spectrum from csv file and calculate spectrum list from 380 to 730 nm.    
    file_name = 'mattila data/'+ lamp_name + '.csv'
    spectrum_df = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
    spectrum = np.zeros(371)
    if spectrum_df.wavelength[0] < 380:
        print('Watch out, spectrum provided have wavelengths lower than 380, this will lead to errors')
    else: 
        offset = int(spectrum_df.wavelength[0]-380)
        max_index = min(len(spectrum_df.wavelength), 370-offset)
        spectrum[offset:offset+max_index] = spectrum_df.intensity[0:max_index]
        
        #normalises spectrum to light intensity
        tot_intensity = spectrum.sum()
        correction_factor = light_intensity / tot_intensity
        spectrum = spectrum * correction_factor * 1e-6

    test_length = 48
    test_light = np.zeros((test_length,371))
    for i in range(test_length):
        test_light[i] = spectrum
    test_time = np.arange(0,5*test_length, 5)/60
    
    # light_intensity = umol m-2 s-1
    #initiate a new_model instance and a results dataframe
    results_frame = run_model(PSI_PSII_ratio=PSI_PSII_ratio_standard,#opt_PS_ratio_mattila, 
                                  dephosphorylation_rate_constant=dephosphorylation_rate_constant_standard, 
                                  phosphorylation_rate_constant=phosphorylation_rate_constant_standard,
                                K_N2 = NPQ_rate,
                                light_list = test_light,
                                model_length=test_length,
                                Time = test_time)


        
    return results_frame.PQH2.iat[-1], spectrum, results_frame, results_frame.D_P700.iat[-1]


def calculate_gaussian_light(center, sigma, intensity):
    #creates a light spectrum based on a gaussian curve. In put is centre wavelength and sigma
    spectrum = np.arange(380, 751, 1)
    probability_density = (1 / math.sqrt(2 * np.pi * (sigma**2))) * np.exp(-0.5 * ((spectrum - center) / sigma) **2)
    spectrum = probability_density * intensity * 1e-6

    return spectrum


def run_mattila_laser_test(wavelength, FWHM, light_intensity):

    #create spectrum with intensity, Full Width at half maximum and wavelength, assume normal distribution
    sigma = FWHM/2.355 #https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    spectrum = calculate_gaussian_light(wavelength, sigma, light_intensity)
    print('laser' + str(spectrum.sum()))

    test_length = 48
    test_light = np.zeros((test_length,371))
    for i in range(test_length):
        test_light[i] = spectrum
    test_time = np.arange(0,5*test_length, 5)/60
    
    # light_intensity = umol m-2 s-1
    #initiate a new_model instance and a results dataframe
    results_frame = run_model(PSI_PSII_ratio=PSI_PSII_ratio_standard,#opt_PS_ratio_mattila, 
                                dephosphorylation_rate_constant=0, 
                                phosphorylation_rate_constant=0,
                                K_N2 = 0,
                                light_list = test_light,
                                model_length=test_length,
                                Time = test_time, 
                                leaf_absorbance=leaf_absorbance_lit)

    return  results_frame.PQH2.iat[-1], spectrum, results_frame, results_frame.D_P700.iat[-1]


results_frames = []
lamp_test_df = pd.DataFrame(columns = ['lamp', 'PQH2_PQ_tot_model', 'spectrum', 'PSII_overex'])
lamp_test_df.lamp = ['halogen', 'incandescent', 'energy saver', 'ww_LED', 'cw_LED', 'mm_halide', 'hp_sodium', 'sun']
for i in range(len(lamp_test_df.lamp)):
    lamp_test_df.iat[i, 1], lamp_test_df.iat[i, 2], result, lamp_test_df.iat[i, 3]= run_mattila_lamp_test(lamp_test_df.lamp[i], 50)
    results_frames.append(result)
    print('progress:' + str(i/len(lamp_test_df.lamp)*100) + '%')

laser_test_df = pd.DataFrame(columns = ['lamp', 'FWHM', 'PQH2_PQ_tot_model', 'spectrum', 'PSII_overex'])
laser_test_df.lamp = [381, 420, 440, 471, 470, 501, 520, 519, 560, 554, 630, 635, 660, 659, 690, 721]
laser_test_df.FWHM = [10, 17, 10, 9, 10, 9, 9, 12, 10, 36, 11, 16, 10, 17, 12, 30]
for i in range(len(laser_test_df.lamp)):
    laser_test_df.iat[i, 2], laser_test_df.iat[i, 3], result, laser_test_df.iat[i, 4]= run_mattila_laser_test(laser_test_df.lamp[i], laser_test_df.FWHM[i], 50)
    results_frames.append(result)
    print('progress:' + str(i/len(laser_test_df.lamp)*100) + '%')
laser_test_df = laser_test_df.astype({"lamp": str})

ordered_list = ['halogen', 'incandescent', 'energy saver', 'ww_LED', 'cw_LED', 'mm_halide', 'hp_sodium', 'sun',
                '381', '420', '440', '471', '470', '501', '520', '519', '560', '554', '630', '635', '660', '659', '690', '721']

experimental = pd.read_csv('mattila data/Mattila_data_all.csv', sep=',', decimal='.', names = ['location', 'PQH2'] )
experimental['index_num'] = experimental.location.apply(lambda x: int(x+1))
#experimental = experimental.groupby('index')['PQH2'].agg(['mean', 'std'])
#experimental = experimental.rename(columns={'mean': 'mean_PQH2', 'std': 'std_PQH2'})
experimental['lamp'] =experimental.index_num.apply(lambda x: ordered_list[x])


model = pd.concat([laser_test_df, lamp_test_df])
lamp_comparison = pd.merge(experimental, model, on='lamp', how='outer')




lamp_comparison['lamp'] = pd.Categorical(lamp_comparison['lamp'], categories=ordered_list, ordered=True)
lamp_comparison = lamp_comparison.sort_values('lamp')
lamp_comparison.PQH2_PQ_tot_model = lamp_comparison.PQH2_PQ_tot_model

fig, axs = plt.subplots(1,1)
for i in range(len(lamp_comparison.location)):
    axs.plot(np.arange(380,751), lamp_comparison.at[i, 'spectrum'], label=lamp_comparison.at[i, 'lamp'])
fig.show()

fig, axs = plt.subplots(1,1)
for i in range(len(lamp_test_df.lamp)):
    axs.plot(np.arange(380,751), lamp_test_df.at[i, 'spectrum'], label=lamp_test_df.at[i, 'lamp'])
fig.show()

#%% find correction factor due to maximum of PQ reduction
NPQ_results_frames = []
NPQ_test_df = pd.DataFrame(columns = ['lamp', 'NPQ', 'NPQ_rate', 'PQH2_PQ_tot_model', 'spectrum', 'PSII_overex'])

NPQ_test_df.NPQ = np.arange(0, 4.01,0.01)
NPQ_test_df.NPQ_rate = NPQ_test_df.NPQ * (K_D2 + K_F2)
NPQ_test_df.lamp = 'halogen'
for i in range(len(NPQ_test_df.lamp)):
    NPQ_test_df.iat[i, 3], NPQ_test_df.iat[i, 4], result, NPQ_test_df.iat[i, 5]= run_mattila_lamp_test(NPQ_test_df.lamp[i], 2000, NPQ_rate=NPQ_test_df.NPQ_rate[i])
    NPQ_results_frames.append(result)
    if i%10==0:
        print('progress:' + str(i/len(NPQ_test_df.lamp)*100) + '%')

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.plot(NPQ_test_df.NPQ, NPQ_test_df.PQH2_PQ_tot_model, ls='-', color='black')
ax.set_xlim(0,4)
ax.set_ylim(0.0,1)
ax.set_xlabel('NPQ')
ax.set_ylabel(r'$PQH_2/PQ_{tot}$')

#%% Draws mattila graphs
def calc_lin_reg(measured, modelled):
    # two lists of equal length are should be provided. 
    # an ordinary least squares regression is used to get an intercept and steepness value. Including two parameters to gauge goodness of fit. 
    x = np.array(measured)
    y = np.array(modelled)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    intercept, steepness = model.params
    r_squared = model.rsquared
    p_value = model.f_pvalue
    return [intercept, steepness, r_squared, p_value]

def calc_RMSD(measured, pred):
    # two lists of equal length should be provided. RMSD between the two lists is returned.
    measured = np.array(measured)
    pred = np.array(pred)
    dif = measured - pred
    dif_squared = dif ** 2
    RMSD = math.sqrt(dif_squared.sum() / len(measured))
    return RMSD


# Map each variable to an index
color_map_index = {v: i for i, v in enumerate(ordered_list)}
# Convert dataframe column to numeric index
color_values = lamp_comparison["lamp"].map(color_map_index)
cmap = plt.get_cmap("viridis", len(ordered_list))

fig, ax = plt.subplots(figsize = (7,5))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)


plot = ax.scatter(lamp_comparison.PQH2, lamp_comparison.PQH2_PQ_tot_model, 
           c=color_values, cmap=cmap)
cbar = plt.colorbar(plot, ticks=range(len(ordered_list)))
cbar.ax.set_yticklabels(ordered_list)
# ax.errorbar(lamp_comparison.PQH2, lamp_comparison.PQH2_PQ_tot_model,
#             xerr=lamp_comparison.std_PQH2, yerr=0, ls='', marker='o', markersize=4)

regression = calc_lin_reg(lamp_comparison.PQH2, lamp_comparison.PQH2_PQ_tot_model)
regression.append(calc_RMSD(lamp_comparison.PQH2, lamp_comparison.PQH2_PQ_tot_model))

ax.plot(np.arange(-0.3, 1.1, 0.1), regression[0] + regression[1] * np.arange(-0.3, 1.1, 0.1), color='blue', label='regression')
ax.plot(np.arange(-0.3, 1.1, 0.1), np.arange(-0.3, 1.1, 0.1), color='grey', label='1:1 line')


string ='$R^2$ = ' + f'{regression[2]:.3f}' + '\n' + 'RMSD = '+ f'{regression[4]:.3f}'  
ax.text(0.97, 0.03, string, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment = 'right', bbox = props)



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'Modelled redox state  ($PQH_2$/$PQ_{tot}$)')
ax.set_xlabel('Measured redox state (Mattila et al. 2020)')  
ax.set_xlim(-0.3, 1)
ax.set_ylim(-0.3, 1)

legend_elements=[]
legend_elements.append(plt.Line2D([0], [0], color='blue', label='regression'))
legend_elements.append(plt.Line2D([0], [0], color='grey', label='measured=modelled'))
plt.legend(handles=legend_elements)



fig.tight_layout()
save_name = os.path.join(output_folder,'Mattila_plot.png')
fig.savefig(save_name,dpi=400)
plt.show()
plt.close()     
#%%
fig, ax = plt.subplots()

for i in range(len(lamp_comparison.lamp)):
    ax.plot(np.arange(380,751, 1), lamp_comparison.spectrum[i]*1e6, label=lamp_comparison.lamp[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('wavelength(nm)')
ax.set_ylabel(r'light intensity [umol $m^{-2}\ s^{-1}\ nm^{-1}$]')  
ax.set_xlim(380, 750)
ax.legend(ncols=4, loc='lower center', bbox_to_anchor=(0.5,1))
#ax.set_ylim(-0.1, 1)

plt.show()       





#%%  Loads incident light spectra Taylor 2019 
#%%initiates Taylor 2019 data


spectrum_SHADE = load_spectrum('SHADE_spectrum', 100) 
spectrum_SUN = load_spectrum('SUN_spectrum', 100)    

fig, ax = plt.subplots()

ax.plot(np.arange(380,751, 1), spectrum_SHADE*1e6, label='shade lamp')
ax.plot(np.arange(380,751, 1), spectrum_SUN*1e6, label='sun lamp')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('wavelength(nm)')
ax.set_ylabel('light intensity umol m-2 s-1 nm-1')  
ax.set_xlim(380, 730)
ax.legend(ncols=1, loc='lower center', bbox_to_anchor=(0.5,1))
#ax.set_ylim(-0.1, 1)

plt.show()

shade_literature = pd.read_csv('model parameters FvCBwd model/SHADE_adapted.csv', sep=';', decimal=',', names=['Time_min', 'Assimilation'])    
sun_literature = pd.read_csv('model parameters FvCBwd model/SUN_adapted.csv', sep=';', decimal=',', names=['Time_min', 'Assimilation']) 

    
spectrum_480 = load_spectrum('480_light', 40) #Correction factor is applied to get 40 umol absorbed photons
spectrum_700_sun = load_spectrum('700_light', 55.6) #Correction factor is applied to get 40 umol absorbed photons        
spectrum_700_shade = load_spectrum('700_light', 53.9) #Correction factor is applied to get 40 umol absorbed photons        


#initialise light and simulation time
section_minutes_Taylor = 65
section_time_Taylor = section_minutes_Taylor * 60
time_steps_model_Taylor = 60
model_length_Taylor =int( 5 * section_time_Taylor / time_steps_model_Taylor)
Ci_model = np.repeat([Ci], model_length_Taylor)
Time_Taylor = np.arange(-1 * section_time_Taylor/60,4 * section_time_Taylor/60, time_steps_model_Taylor/60)

t_plot_Taylor = (0, 4*section_time_Taylor/60)



    
light_list_Taylor_sun = np.zeros((model_length_Taylor,371))
for i in range(len(light_list_Taylor_sun)):
    if (i>=len(light_list_Taylor_sun)/5) and (i<len(light_list_Taylor_sun)*2/5):
        light_list_Taylor_sun[i] = spectrum_480
    elif (i>=len(light_list_Taylor_sun)*3/5) and (i<len(light_list_Taylor_sun)*4/5):
        light_list_Taylor_sun[i] = spectrum_480
    else:
        light_list_Taylor_sun[i] = spectrum_700_sun

light_list_Taylor_shade = np.zeros((model_length_Taylor,371))
for i in range(len(light_list_Taylor_shade)):
    if (i>=len(light_list_Taylor_shade)/5) and (i<len(light_list_Taylor_shade)*2/5):
        light_list_Taylor_shade[i] = spectrum_480
    elif (i>=len(light_list_Taylor_shade)*3/5) and (i<len(light_list_Taylor_shade)*4/5):
        light_list_Taylor_shade[i] = spectrum_480
    else:
        light_list_Taylor_shade[i] = spectrum_700_shade

#Running tests for Taylor data
def run_taylor_test(light, PS_ratio, dephos_rate, phos_rate):
    results_frame = run_model(PSI_PSII_ratio=PS_ratio, 
                              dephosphorylation_rate_constant=dephos_rate, 
                              phosphorylation_rate_constant=phos_rate,
                              K_N2 = 0,
                              light_list = light,
                              model_length=model_length_Taylor,
                              Time = Time_Taylor,
                              cutoff=5,
                              Rd=0,
                              time_steps_model=time_steps_model_Taylor, 
                              leaf_absorbance=leaf_absorbance_NT)
    return results_frame


#%% runs taylor 2019 tests
dephos_list = np.exp(np.arange(-7,-4, 0.5 ))
phos_ratio_list = np.arange(1.0,2.0,0.2)
phos_list = np.outer(dephos_list, phos_ratio_list)#np.exp(np.arange(-6.5,-4.5, 0.2 ))
error = np.zeros((len(dephos_list), len(phos_list[0])))
for i in range(len(dephos_list)):
    print(f'progress: {(i)/len(dephos_list)*100:.0f}%')
    for j in range(len(phos_list[0])):
        
        assimilation_SHADE, opt_PS_ratio_SHADE = find_opt_PS_ratio(spectrum_SHADE, phos_rate=phos_list[i,j],
                                                                   dephos_rate=dephos_list[i])   
        assimilation_SUN, opt_PS_ratio_SUN = find_opt_PS_ratio(spectrum_SUN, phos_rate=phos_list[i,j],
                                                               dephos_rate=dephos_list[i])   
        
        results_frame_shade = run_taylor_test(light_list_Taylor_shade, 0.5, dephos_list[i], phos_list[i,j])
        results_frame_sun = run_taylor_test(light_list_Taylor_sun, 0.6, dephos_list[i], phos_list[i,j])
        
        error_shade = calc_RMSD(shade_literature.Assimilation[:-1], results_frame_shade.A)
        error_sun = calc_RMSD(sun_literature.Assimilation[:-1], results_frame_sun.A)
        total_error = error_shade + error_sun
        error[i,j] = total_error

loc_min = np.argmin(error)
opt_phos_rate = phos_list[loc_min//len(phos_list[0]), loc_min%len(phos_list[0])]
opt_dephos_rate = dephos_list[loc_min//len(phos_list[0])]
# %%Calculates taylor 2019 fit
phos_taylor = opt_phos_rate
dephos_taylor = opt_dephos_rate

assimilation_SHADE, opt_PS_ratio_SHADE = find_opt_PS_ratio(spectrum_SHADE, phos_rate=phos_taylor,
                                                           dephos_rate=dephos_taylor, plot_figures=True)   
assimilation_SUN, opt_PS_ratio_SUN = find_opt_PS_ratio(spectrum_SUN, phos_rate=phos_taylor,
                                                           dephos_rate=dephos_taylor, plot_figures=True)   

results_frame_shade = run_taylor_test(light_list_Taylor_shade, 0.5, dephos_taylor, phos_taylor)
results_frame_sun = run_taylor_test(light_list_Taylor_sun, 0.6, dephos_taylor, phos_taylor)


# Plots  results
fig, ax = plt.subplots()

ax.plot(shade_literature.Time_min, shade_literature.Assimilation, label='Data Taylor et al. SHADE',
        linestyle='--', color='blue', alpha=0.5)
ax.plot(sun_literature.Time_min, sun_literature.Assimilation, label='Data Taylor et al. SUN',
        linestyle='--', color='red', alpha=0.5)

ax.plot(results_frame_shade.Time, results_frame_shade.A, label='SHADE modelled', linestyle='-', color='blue', alpha=0.5)
ax.plot(results_frame_sun.Time, results_frame_sun.A, label='SUN modelled', linestyle='-', color='red', alpha=0.5)



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Time (min)')
ax.set_ylabel(r'Assimilation [$\mu mol\ m^{-2}\ s^{-1}$]')  

measured_line = Line2D([],[], c='black', ls='--' )
modelled_line = Line2D([],[], c='black', ls='-' )
shade_patch = Patch(facecolor='blue', alpha=0.5)
sun_patch = Patch(facecolor='red', alpha=0.5)


ax.legend(
    [measured_line, shade_patch, modelled_line, sun_patch],
    ['Measured assimilation', 'Shade adapted leaf', 'Modelled assimilation', 'Sun adapted leaf'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc='upper center', bbox_to_anchor=(0.5, 0.95),
    ncols=2
)


ax.set_xlim(0,260)
ax.set_ylim(1.8, 3.4)


   
colors = ['lightskyblue', 'lightcoral','lightskyblue', 'lightcoral']
texts = ['Blue LED', 'FR LED','Blue LED', 'FR LED']


for k in range(4):
    
    banner = mpatches.Rectangle((0+k*0.25, 0.94), 0.25+k*0.25, 0.06,
                                transform=ax.transAxes,  # axes coords
                                facecolor=colors[k],
                                edgecolor=colors[k], linewidth=0.8)
    ax.add_patch(banner)
    ax.text(0.25/2 + k*0.25, 0.975, texts[k], transform=ax.transAxes,
            va="center_baseline", ha="center", fontsize=10, color="black")
    ax.vlines(section_minutes_Taylor*k,-10,150, ls='--', color='lightgrey', zorder=0)



fig.tight_layout()
save_name = os.path.join(output_folder,'Taylor_data_fit.png')
fig.savefig(save_name,dpi=400)
plt.show()
plt.close() 
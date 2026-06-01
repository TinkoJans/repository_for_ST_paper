# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:55:11 2025

@author: jans008
"""

#%%
#step function for calculating the next time step in ST analysis. 
# It takes in all the state parameters of the previous timestep, 
# individual parameters for that timestep such as Ci and light_spectrum
# and it takes general model parameters such as: Rd, Vcmax or phosphorylation rate constant.

#Finally it returns updates state variables. 
def take_time_step (Ci, light_spect,  A_prev, chloroplastic_carbon, LHCII_per_PSII, 
                    LHCII_per_PSI, D_P680, D_P700, PQH2, Fm, Fm_2, Fo, Fs,
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
 
    D_P680_abs = pm.math.where(D_P700_abs > D_P700_tot,
                       ((Phi_P700_photo * P700_excitation) / eta) * ((1 - phi_U2) / (Phi_P680_photo * P680_excitation)) * D_P680_tot,
                       D_P680_abs)
    PQH2_abs = pm.math.where(D_P700_abs > D_P700_tot, (1 - (D_P680_abs / D_P680_tot)) * PQ_pool, PQH2_abs)
    D_P700_abs = pt.minimum(D_P700_tot, D_P700_abs)
    
    #calc_J_P680_forward()
    Phi_P680_photo = K_P2 / rate_tot_P680_open
    J_P680_forward = D_P680_abs * Phi_P680_photo * P680_excitation / (D_P680_tot * (1 - phi_U2))

    #calc_J_P700_forward()
    Phi_P700_photo = K_P1 / rate_tot_P700_open
    J_P700_forward = D_P700_abs * Phi_P700_photo * P700_excitation / D_P700_tot 


    W_J = (chloroplastic_carbon * J_P680_forward) / (4 * chloroplastic_carbon + 8 * gamma_star)
    W_c = (chloroplastic_carbon * Vcmax) / (chloroplastic_carbon + MM_constant_carboxy * (1 + (chloroplastic_oxygen / MM_constant_oxy)))
    A = (1 - (gamma_star / chloroplastic_carbon)) * pt.minimum(W_c, W_J) - Rd
    chloroplastic_carbon = Ci - (A / g_meso)
    
    
    #updating LHCII distribution
    LHCII_dephosphorylated = LHCII_per_PSII * D_P680_tot 
    LHCII_phosphorylated = LHCII_per_PSI * D_P700_tot 
    dephosphorylation_rate = LHCII_phosphorylated * dephosphorylation_rate_constant
    phosphorylation_rate = LHCII_dephosphorylated * phosphorylation_rate_constant * (PQH2_abs / PQ_pool)
    
    change_rate = (phosphorylation_rate - dephosphorylation_rate) * timestep
    LHCII_phosphorylated += change_rate 
    LHCII_dephosphorylated -= change_rate 
    
        
    LHCII_per_PSII_new = LHCII_dephosphorylated / D_P680_tot
    LHCII_per_PSI_new = LHCII_phosphorylated / D_P700_tot
    
    D_P680 = D_P680_abs / D_P680_tot
    D_P700 = D_P700_abs / D_P700_tot
    PQH2 = PQH2_abs / PQ_pool
    
    #calculations for qT
    excitation_PSII_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_II).sum(axis=0)
    excitation_PSI_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_I).sum(axis=0)
    
    a_m = excitation_PSII_PAM * epsilon_F2 * (K_F2 / (rate_tot_P680_closed-K_U2))
    b_m = excitation_PSI_PAM * epsilon_F1 * (K_F1 / rate_tot_P700_closed)
    Fm = a_m + b_m
    Fm_2 = a_m
    
    
    #calculations for qL
    #Fo

    a_o = excitation_PSII_PAM * epsilon_F2 * (K_F2 / (rate_tot_P680_open-K_U2))
    b_o = excitation_PSI_PAM * epsilon_F1 * (K_F1 / rate_tot_P700_open)
    Fo = a_o + b_o

    #Fs
    Fs = a_o * D_P680 + a_m * (1-D_P680) + b_o * D_P700 + b_m + (1-D_P700)

    
    
    
    

    
    new_state = (A, chloroplastic_carbon, LHCII_per_PSII_new, LHCII_per_PSI_new, D_P680, D_P700, PQH2, Fm, Fm_2, Fo, Fs, P680_excitation, P700_excitation, J_P680_forward, J_P700_forward)

    return new_state
    #%%Loading st data
def solve_for_log_normal_parameters(mean, sigma):
    variance = np.log(sigma**2/mean**2 + 1)
    mu = np.log(mean) - variance/2
    return (mu, variance**0.5)

def run_model():
    
    with basic_model:
        # Priors for unknown model parameters
        Vcmax_params = solve_for_log_normal_parameters(prior_table.at['Vcmax', 'mean']*1e-6, 
                                                       prior_table.at['Vcmax', 'std']*1e-6)
        Vcmax = pm.Normal('Vcmax', mu=Vcmax_params[0], sigma=Vcmax_params[1])

        #Chl_per_area
        Chl_per_area_params = solve_for_log_normal_parameters(prior_table.at['Chl_per_area', 'mean'], 
                                                       prior_table.at['Chl_per_area', 'std'])
        Chl_per_area = pm.Normal('Chl_per_area', mu=Chl_per_area_params[0], sigma=Chl_per_area_params[1])
        
        #eta_LET
        eta_LET_params = solve_for_log_normal_parameters(prior_table.at['eta_LET', 'mean'], 
                                                       prior_table.at['eta_LET', 'std'])
        eta_LET = pm.Normal('eta_LET', mu=eta_LET_params[0], sigma=eta_LET_params[1])
        
        #eta_CET
        eta_CET_params = solve_for_log_normal_parameters(prior_table.at['eta_CET', 'mean'], 
                                                       prior_table.at['eta_CET', 'std'])
        eta_CET = pm.Normal('eta_CET', mu=eta_CET_params[0], sigma=eta_CET_params[1])
        
        #gamma_star
        gamma_star_params = solve_for_log_normal_parameters(prior_table.at['gamma_star', 'mean']*1e-6, 
                                                       prior_table.at['gamma_star', 'std']*1e-6)
        gamma_star = pm.Normal('gamma_star', mu=gamma_star_params[0], sigma=gamma_star_params[1])
        
        #K_P2           
        K_P2_params = solve_for_log_normal_parameters(prior_table.at['K_p2', 'mean'], 
                                                       prior_table.at['K_p2', 'std'])
        K_P2 = pm.Normal('K_P2', mu=K_P2_params[0], sigma=K_P2_params[1])
        
        #K_P1
        K_P1_params = solve_for_log_normal_parameters(prior_table.at['K_p1', 'mean'], 
                                                       prior_table.at['K_p1', 'std'])
        K_P1 = pm.Normal('K_P1', mu=K_P1_params[0], sigma=K_P1_params[1])
        
        #PS ratio
        PSI_PSII_ratio_params = solve_for_log_normal_parameters(prior_table.at['PSI_PSII_ratio', 'mean'], 
                                                       prior_table.at['PSI_PSII_ratio', 'std'])
        PSI_PSII_ratio_sun = pm.Normal('PSI_PSII_ratio_sun', mu=PSI_PSII_ratio_params[0], sigma=PSI_PSII_ratio_params[1])
        PSI_PSII_ratio_shade = pm.Normal('PSI_PSII_ratio_shade', mu=PSI_PSII_ratio_params[0], sigma=PSI_PSII_ratio_params[1])
        
        
        #PQ_per_chl
        PQ_per_chl_params = solve_for_log_normal_parameters(prior_table.at['PQ_per_chl', 'mean'], 
                                                      prior_table.at['PQ_per_chl', 'std'])
        PQ_per_chl = pm.Normal('PQ_per_chl', mu=PQ_per_chl_params[0], sigma=PQ_per_chl_params[1])
        
        #MM_constant_oxy
        MM_constant_oxy_params = solve_for_log_normal_parameters(prior_table.at['MM_constant_oxy', 'mean']*1e-3, 
                                                       prior_table.at['MM_constant_oxy', 'std']*1e-3)
        MM_constant_oxy = pm.Normal('MM_constant_oxy', mu=MM_constant_oxy_params[0], sigma=MM_constant_oxy_params[1])
        
        #MM_constant_carboxy
        MM_constant_carboxy_params = solve_for_log_normal_parameters(prior_table.at['MM_constant_carboxy', 'mean']*1e-6, 
                                                       prior_table.at['MM_constant_carboxy', 'std']*1e-6)
        MM_constant_carboxy = pm.Normal('MM_constant_carboxy', mu=MM_constant_carboxy_params[0], sigma=MM_constant_carboxy_params[1])
        

        #PS_per_chl
        PS_per_chl_params = solve_for_log_normal_parameters(prior_table.at['PS_per_chl', 'mean'], 
                                                       prior_table.at['PS_per_chl', 'std'])
        PS_per_chl = pm.Normal('PS_per_chl', mu=PS_per_chl_params[0], sigma=PS_per_chl_params[1])
        
        #K_N2
        K_N2_params = solve_for_log_normal_parameters(prior_table.at['K_N2', 'mean'], 
                                                       prior_table.at['K_N2', 'std'])
        K_N2 = pm.Normal('K_N2', mu=K_N2_params[0], sigma=K_N2_params[1])
        
        #alpha leaf
        leaf_absorbance_change = pm.Normal('leaf_abs_change', mu=0, sigma=0.05)
        
        #g_meso
        g_meso_params = solve_for_log_normal_parameters(prior_table.at['g_m', 'mean'], 
                                                       prior_table.at['g_m', 'std'])
        g_meso = pm.Normal('g_meso', mu=g_meso_params[0], sigma=g_meso_params[1])
        
        #CB6F_per_chl
        CB6F_per_chl_params = solve_for_log_normal_parameters(prior_table.at['CB6F_per_chl', 'mean'], 
                                                       prior_table.at['CB6F_per_chl', 'std'])
        CB6F_per_chl = pm.Normal('CB6F_per_chl', mu=CB6F_per_chl_params[0], sigma=CB6F_per_chl_params[1])
        
        #g_meso
        k_max_CB6F_params = solve_for_log_normal_parameters(prior_table.at['k_max_CB6F', 'mean'], 
                                                       prior_table.at['k_max_CB6F', 'std'])
        k_max_CB6F = pm.Normal('k_max_CB6F', mu=k_max_CB6F_params[0], sigma=k_max_CB6F_params[1])
        
        #Phosphorylation and dephosphorylation
        phos_params = solve_for_log_normal_parameters(0.001, 0.001)
        phosphorylation_rate_constant = pm.Normal(name='forward_ST_rate', mu=phos_params[0], sigma=phos_params[1])
        
        dephos_params = solve_for_log_normal_parameters(0.001, 0.001)
        dephosphorylation_rate_constant = pm.Normal(name='reverse_ST_rate', mu=dephos_params[0], sigma=dephos_params[1])
        
        #chl_a_b Taken from Taylor 2019
        chl_ab_params_sun = solve_for_log_normal_parameters(3.08, 0.02)
        chl_ab_sun = pm.Normal('chl_ab_sun', mu=chl_ab_params_sun[0], sigma=chl_ab_params_sun[1])
        
        chl_ab_params_shade = solve_for_log_normal_parameters(2.89, 0.02)
        chl_ab_shade = pm.Normal('chl_ab_shade', mu=chl_ab_params_shade[0], sigma=chl_ab_params_shade[1])
        
        #remaining error
        sigma_params = solve_for_log_normal_parameters(0.1e-6, 0.1e-6)
        sigma = pm.Normal('sigma', mu=sigma_params[0], sigma=sigma_params[1])
        
        #defining ST model parameters
      
        rate_tot_P680_open = pt.exp(K_P2) + pt.exp(K_N2) + pt.exp(K_D2) + K_F2 + K_U2
        rate_tot_P680_closed = pt.exp(K_N2) + pt.exp(K_D2) + K_F2 + K_U2
        open_sharing_rate = K_U2 / rate_tot_P680_open
        closed_sharing_rate = K_U2 / rate_tot_P680_closed
        
        K_X1 = pt.exp(K_P1) # ns-1, rate of dissipation in closed P700, set equal to K_P1
        rate_tot_P700_open = pt.exp(K_P1) + K_D1 + K_F1
        rate_tot_P700_closed = K_X1 + K_D1 + K_F1
        
        #initialise thylakoid membrane
        global PQ_pool
        PQ_pool = pt.exp(PQ_per_chl) * pt.exp(Chl_per_area) # mol m-2


        leaf_absorbance = leaf_absorbance_measured + leaf_absorbance_change
        
        #initalise initial state
        A = 0
        chloroplastic_carbon_init = Ci_model[0]

        D_P680_init = 1 #D_P680 is initiated fully open
        D_P700_init = 1 #D_P680 is initiated fully open
        PQH2_init = 0
        Fm_init = 1 # placeholder for Fm, value is not used
        Fm_2_init = 1 # placeholder for Fm, value is not used
        Fo_init = 1 # placeholder for Fo, value is not used
        Fs_init = 1 # placeholder for Fs, value is not used
        P680_excitation_init = 40 # placeholder for excitation, value is not used
        P700_excitation_init = 40 # placeholder for excitation, value is not used
        J_P680_forward_init = 40 # placeholder for flow rate, value is not used
        J_P700_forward_init = 40 # placeholder for flow rate, value is not used

 #_______________________Sun Code_______________________________________       
        concentration_PSII_sun = pt.exp(PS_per_chl) * (1 - (pt.exp(PSI_PSII_ratio_sun) / (1 + pt.exp(PSI_PSII_ratio_sun)))) # mol/mol Chl
        concentration_PSI_sun = pt.exp(PS_per_chl) - concentration_PSII_sun # mol/mol Chl
        global D_P680_tot_sun
        D_P680_tot_sun = concentration_PSII_sun * pt.exp(Chl_per_area) # mol m-2
        global D_P700_tot_sun
        D_P700_tot_sun = concentration_PSI_sun * pt.exp(Chl_per_area) # mol m-2
        
        #calculate LHCII amount from PSI/PSII ratio and from chl_ab ratio
        a_sun = (PS_table.at['PSI-LHCI', 'chl_a'] - pt.exp(chl_ab_sun) * PS_table.at['PSI-LHCI', 'chl_b']) / (1 + (1/pt.exp(PSI_PSII_ratio_sun)))
        b_sun = (PS_table.at['CSM', 'chl_a'] - pt.exp(chl_ab_sun) * PS_table.at['CSM', 'chl_b'])/ (1 + pt.exp(PSI_PSII_ratio_sun))
        c_sun = pt.exp(chl_ab_sun) * PS_table.at['LHCII', 'chl_b'] - PS_table.at['LHCII', 'chl_a']
        extra_LHCII_per_PS_sun = (a_sun + b_sun) / c_sun
        
        LHCII_per_PSII_init_sun = 2 + extra_LHCII_per_PS_sun #LHCII_per_PSII_init
        LHCII_per_PSI_init_sun = extra_LHCII_per_PS_sun #LHCII_per_PSI_init


        initial_state_sun = [
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(chloroplastic_carbon_init, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSII_init_sun, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSI_init_sun, dtype='float64'),
            pt.as_tensor_variable(D_P680_init, dtype='float64'),
            pt.as_tensor_variable(D_P700_init, dtype='float64'),
            pt.as_tensor_variable(PQH2_init, dtype='float64'),
            pt.as_tensor_variable(Fm_init, dtype='float64'),
            pt.as_tensor_variable(Fm_2_init, dtype='float64'),
            pt.as_tensor_variable(Fo_init, dtype='float64'),
            pt.as_tensor_variable(Fs_init, dtype='float64'),
            pt.as_tensor_variable(P680_excitation_init, dtype='float64'),
            pt.as_tensor_variable(P700_excitation_init, dtype='float64'),
            pt.as_tensor_variable(J_P680_forward_init, dtype='float64'),
            pt.as_tensor_variable(J_P700_forward_init, dtype='float64'),
        ]
        
        
        #make tensors out of input parameters
        Ci = pt.as_tensor_variable(Ci_model)
        light_spect_sun = pt.as_tensor_variable(light_lists[0])

        #revert to non transformed variables
        dephosphorylation_rate_constant = pt.exp(dephosphorylation_rate_constant)
        phosphorylation_rate_constant = pt.exp(phosphorylation_rate_constant)
        Vcmax = pt.exp(Vcmax)
        Rd =0
        eta_LET = pt.exp(eta_LET)
        eta_CET = pt.exp(eta_CET)
        gamma_star = pt.exp(gamma_star)
        MM_constant_oxy = pt.exp(MM_constant_oxy)
        MM_constant_carboxy = pt.exp(MM_constant_carboxy)
        CB6F_recovery_rate = pt.exp(CB6F_per_chl) * pt.exp(k_max_CB6F) * pt.exp(Chl_per_area)
        g_meso = pt.exp(g_meso)
        K_P2 = pt.exp(K_P2)
        K_P1 = pt.exp(K_P1)

            
        """This function loops over the complete timerange that is modelled. It takes
        an initial state and parameters needed for running the model. Ci and light_spect are different for each
        data point and are thus a list. THe other parameters are constant and are single values. """
        results_sun, _ = pytensor.scan(
            fn = take_time_step,
            outputs_info=[{"initial": s, "taps": [-1]} for s in initial_state_sun],
            sequences = [Ci[:], light_spect_sun[:]],
            non_sequences=[dephosphorylation_rate_constant, phosphorylation_rate_constant,
                           Vcmax, Rd, eta_LET, eta_CET, gamma_star,
                           MM_constant_oxy, MM_constant_carboxy,
                           CB6F_recovery_rate, g_meso, K_P2, K_P1,
                           D_P680_tot_sun, D_P700_tot_sun, PQ_pool, 
                           rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
                           open_sharing_rate, closed_sharing_rate, leaf_absorbance,
                           time_steps_model],
            n_steps = timeseries_length,
            strict=True)
            
        #Defines which parameters will be provided as output.
        A_sun = pm.Deterministic("A_sun", results_sun[0], dims=("timeseries_length",))
        Cc_sun = pm.Deterministic("Cc_sun", results_sun[1], dims=("timeseries_length",))
        LHCII_per_PSII_sun = pm.Deterministic("LHCII_per_PSII_sun", results_sun[2], dims=("timeseries_length",))
        LHCII_per_PSI_sun = pm.Deterministic("LHCII_per_PSI_sun", results_sun[3], dims=("timeseries_length",))
        D_P680_sun = pm.Deterministic("D_P680_sun", results_sun[4], dims=("timeseries_length",))
        D_P700_sun = pm.Deterministic("D_P700_sun", results_sun[5], dims=("timeseries_length",))
        PQH2_sun = pm.Deterministic("PQH2_sun", results_sun[6], dims=("timeseries_length",))
        Fm_sun = pm.Deterministic("Fm_sun", results_sun[7], dims=("timeseries_length",))
        Fm_2_sun = pm.Deterministic("Fm_2_sun", results_sun[8], dims=("timeseries_length",))
        Fo_sun = pm.Deterministic("Fo_sun", results_sun[9], dims=("timeseries_length",))
        Fs_sun = pm.Deterministic("Fs_sun", results_sun[10], dims=("timeseries_length",))
        P680_excitation_sun = pm.Deterministic("P680_excitation_sun", results_sun[11], dims=("timeseries_length",))
        P700_excitation_sun = pm.Deterministic("P700_excitation_sun", results_sun[12], dims=("timeseries_length",))
        J_P680_forward_sun = pm.Deterministic("J_P680_forward_sun", results_sun[13], dims=("timeseries_length",))
        J_P700_forward_sun = pm.Deterministic("J_P700_forward_sun", results_sun[14], dims=("timeseries_length",))
        
        Fm_ratio_sun = Fm_sun/Fm_sun[index_Fm]
        Fm_saved_sun = pm.Deterministic("Fm_saved_sun", Fm_ratio_sun, dims=("timeseries_length",))
        
        Fm_mod_sun = Fm_ratio_sun[np.array(Fm_obs_masks[0])]
        # Determines subset of modelled A values that are also observed. 
        A_obs_sun = A_sun[np.array(obs_mask)]
        # Likelihood (sampling distribution) of observations
        Y_obs_sun = pm.Normal("Y_obs_sun", mu=A_obs_sun, sigma=pt.exp(sigma), observed=literature_assimilation[0])
        #Fm_obs_sun = pm.Normal('Fm_obs_sun', mu=Fm_mod_sun, sigma = pt.exp(sigma_qT), observed = np.array(literature_Fm[0]))
        
        #_______________________Shade COde_______________________________________       
        concentration_PSII_shade = pt.exp(PS_per_chl) * (1 - (pt.exp(PSI_PSII_ratio_shade) / (1 + pt.exp(PSI_PSII_ratio_shade)))) # mol/mol Chl
        concentration_PSI_shade = pt.exp(PS_per_chl) - concentration_PSII_shade # mol/mol Chl
        global D_P680_tot_shade
        D_P680_tot_shade = concentration_PSII_shade * pt.exp(Chl_per_area) # mol m-2
        global D_P700_tot_shade
        D_P700_tot_shade = concentration_PSI_shade * pt.exp(Chl_per_area) # mol m-2
       
        #calculate LHCII amount from PSI/PSII ratio and from chl_ab ratio
        a_shade = (PS_table.at['PSI-LHCI', 'chl_a'] - pt.exp(chl_ab_shade) * PS_table.at['PSI-LHCI', 'chl_b']) / (1 + (1/pt.exp(PSI_PSII_ratio_shade)))
        b_shade = (PS_table.at['CSM', 'chl_a'] - pt.exp(chl_ab_shade) * PS_table.at['CSM', 'chl_b'])/ (1 + pt.exp(PSI_PSII_ratio_shade))
        c_shade = pt.exp(chl_ab_shade) * PS_table.at['LHCII', 'chl_b'] - PS_table.at['LHCII', 'chl_a']
        extra_LHCII_per_PS_shade = (a_shade + b_shade) / c_shade
        
        LHCII_per_PSII_init_shade = 2 + extra_LHCII_per_PS_shade #LHCII_per_PSII_init
        LHCII_per_PSI_init_shade = extra_LHCII_per_PS_shade #LHCII_per_PSI_init


        initial_state_shade = [
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(chloroplastic_carbon_init, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSII_init_shade, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSI_init_shade, dtype='float64'),
            pt.as_tensor_variable(D_P680_init, dtype='float64'),
            pt.as_tensor_variable(D_P700_init, dtype='float64'),
            pt.as_tensor_variable(PQH2_init, dtype='float64'),
            pt.as_tensor_variable(Fm_init, dtype='float64'),
            pt.as_tensor_variable(Fm_2_init, dtype='float64'),
            pt.as_tensor_variable(Fo_init, dtype='float64'),
            pt.as_tensor_variable(Fs_init, dtype='float64'),
            pt.as_tensor_variable(P680_excitation_init, dtype='float64'),
            pt.as_tensor_variable(P700_excitation_init, dtype='float64'),
            pt.as_tensor_variable(J_P680_forward_init, dtype='float64'),
            pt.as_tensor_variable(J_P700_forward_init, dtype='float64'),
        ]
        
        
        #make tensors out of input parameters
        light_spect_shade = pt.as_tensor_variable(light_lists[1])
 

        """This function loops over the complete timerange that is modelled. It takes
        an initial state and parameters needed for running the model. Ci and light_spect are different for each
        data point and are thus a list. THe other parameters are constant and are single values. """
        results_shade, _ = pytensor.scan(
            fn = take_time_step,
            outputs_info=[{"initial": s, "taps": [-1]} for s in initial_state_shade],
            sequences = [Ci[:], light_spect_shade[:]],
            non_sequences=[dephosphorylation_rate_constant, phosphorylation_rate_constant,
                           Vcmax, Rd, eta_LET, eta_CET, gamma_star,
                           MM_constant_oxy, MM_constant_carboxy,
                           CB6F_recovery_rate, g_meso, K_P2, K_P1,
                           D_P680_tot_shade, D_P700_tot_shade, PQ_pool, 
                           rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
                           open_sharing_rate, closed_sharing_rate, leaf_absorbance,
                           time_steps_model],
            n_steps = timeseries_length,
            strict=True)
            
        #Defines which parameters will be provided as output.
        A_shade = pm.Deterministic("A_shade", results_shade[0], dims=("timeseries_length",))
        Cc_shade = pm.Deterministic("Cc_shade", results_shade[1], dims=("timeseries_length",))
        LHCII_per_PSII_shade = pm.Deterministic("LHCII_per_PSII_shade", results_shade[2], dims=("timeseries_length",))
        LHCII_per_PSI_shade = pm.Deterministic("LHCII_per_PSI_shade", results_shade[3], dims=("timeseries_length",))
        D_P680_shade = pm.Deterministic("D_P680_shade", results_shade[4], dims=("timeseries_length",))
        D_P700_shade = pm.Deterministic("D_P700_shade", results_shade[5], dims=("timeseries_length",))
        PQH2_shade = pm.Deterministic("PQH2_shade", results_shade[6], dims=("timeseries_length",))
        Fm_shade = pm.Deterministic("Fm_shade", results_shade[7], dims=("timeseries_length",))
        Fm_2_shade = pm.Deterministic("Fm_2_shade", results_shade[8], dims=("timeseries_length",))
        Fo_shade = pm.Deterministic("Fo_shade", results_shade[9], dims=("timeseries_length",))
        Fs_shade = pm.Deterministic("Fs_shade", results_shade[10], dims=("timeseries_length",))
        P680_excitation_shade = pm.Deterministic("P680_excitation_shade", results_shade[11], dims=("timeseries_length",))
        P700_excitation_shade = pm.Deterministic("P700_excitation_shade", results_shade[12], dims=("timeseries_length",))
        J_P680_forward_shade = pm.Deterministic("J_P680_forward_shade", results_shade[13], dims=("timeseries_length",))
        J_P700_forward_shade = pm.Deterministic("J_P700_forward_shade", results_shade[14], dims=("timeseries_length",))
        
        Fm_ratio_shade = Fm_shade/Fm_shade[index_Fm]
        Fm_saved_shade = pm.Deterministic("Fm_saved_shade", Fm_ratio_shade, dims=("timeseries_length",))
        
        Fm_mod_shade = Fm_ratio_shade[np.array(Fm_obs_masks[1])]
        # Determines subset of modelled A values that are also observed. 
        A_obs_shade = A_shade[np.array(obs_mask)]
        # Likelihood (sampling distribution) of observations
        Y_obs_shade = pm.Normal("Y_obs_shade", mu=A_obs_shade, sigma=pt.exp(sigma), observed=literature_assimilation[1])
        #Fm_obs_shade = pm.Normal('Fm_obs_shade', mu=Fm_mod_shade, sigma = pt.exp(sigma_qT), observed = np.array(literature_Fm[1]))
      #_______________________________________________________________________________
        #do some prior predictive checks. Does the model make sense with the selected priors?
        global idata
        idata = pm.sample_prior_predictive(draws=50, random_seed=rng)
        global prior
        prior = idata.prior
          
        #plots prior predictive checks
        fig, ax = plt.subplots(2,2, figsize=(6,5))
        az.plot_hdi(Time_obs, idata.prior_predictive['Y_obs_sun'], ax=ax[0,0])
        #az.plot_hdi(Times_Fm[0], idata.prior_predictive['Fm_obs_sun'], ax=ax[0,1])
        
        az.plot_hdi(Time_obs, idata.prior_predictive['Y_obs_shade'], ax=ax[1,0])
        #az.plot_hdi(Times_Fm[1], idata.prior_predictive['Fm_obs_shade'], ax=ax[1,1])
        Y_obs_samples_sun = idata.prior_predictive['Y_obs_sun'].values
        #Fm_obs_samples_sun = idata.prior_predictive['Fm_obs_sun'].values
        Y_obs_samples_shade = idata.prior_predictive['Y_obs_shade'].values
        #Fm_obs_samples_shade = idata.prior_predictive['Fm_obs_shade'].values
        
        fig, ax = plt.subplots(2,2, figsize=(8,8))
        # for i in range(Fm_obs_samples_sun.shape[1]):
        #     ax[0,0].plot(Times_Fm[0], Fm_obs_samples_sun[0,i,:], alpha=0.1, c='k')
        #ax[0,0].scatter(Times_Fm[0], idata.observed_data['Fm_obs_sun'])
        ax[0,0].set_xlabel('Time')
        ax[0,0].set_ylabel('qT predicted')
        ax[0,0].set_title('Prior predictive checks')
        
        for i in range(Y_obs_samples_sun.shape[1]):
           ax[0,1].plot(Time_obs, Y_obs_samples_sun[0,i,:], alpha=0.1, c='k')
        ax[0,1].scatter(Time_obs, idata.observed_data['Y_obs_sun'])
        ax[0,1].set_xlabel('Time')
        ax[0,1].set_ylabel('assimilation predicted')
        ax[0,1].set_title('Prior predictive checks')
        
        # for i in range(Fm_obs_samples_shade.shape[1]):
        #     ax[1,0].plot(Times_Fm[1], Fm_obs_samples_shade[0,i,:], alpha=0.1, c='k')
        #ax[1,0].scatter(Times_Fm[1], idata.observed_data['Fm_obs_shade'])
        ax[1,0].set_xlabel('Time')
        ax[1,0].set_ylabel('qT predicted')
        ax[1,0].set_title('Prior predictive checks')
        
        for i in range(Y_obs_samples_shade.shape[1]):
           ax[1,1].plot(Time_obs, Y_obs_samples_shade[0,i,:], alpha=0.1, c='k')
        ax[1,1].scatter(Time_obs, idata.observed_data['Y_obs_shade'])
        ax[1,1].set_xlabel('Time')
        ax[1,1].set_ylabel('assimilation predicted')
        ax[1,1].set_title('Prior predictive checks')
        
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'prior_predictive.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 

        global posterior
    
        n_chains = 3
        #Option 2, let the laplace approx change the model
        # seems much faster, but unclear if result is the same. 
        if False:
            with basic_model:
                #Run laplacian approximation
                idata_approx = pymc_extras.inference.fit_laplace()

    
        # Run sampling analysiss
        if True:
            n_chains = 10
            idata = pm.sample(cores=n_chains, chains=n_chains, tune=1000, draws=1000,
                              progressbar=True,return_inferencedata=True,
                              init="jitter+adapt_diag", target_accept=0.9) 

        else:
            idata = idata_approx
    return idata, basic_model

#%% This code needs to be used when using multiple cores. 

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
    from initial_data_processing import plant_instance
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from scipy.stats import lognorm
    from scipy.stats import norm
    import matplotlib.lines as mlines
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple


    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    base_directory = os.path.dirname(os.path.realpath(__file__))
    
    def find_or_create_results_folder(plantID):
        folder_name = 'theoretical figures'
        folder_location = os.path.join(base_directory, folder_name, plantID)
        if os.path.isdir(folder_location):
            return folder_location
        else:
            os.mkdir(folder_location)
            return folder_location
    
    # Initialize random number generator
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    #az.style.use("arviz-darkgrid")
    
    
    chloroplastic_oxygen = 212.7*1e-3*(2/21) # bar, 21% of atmospheric pressure
        
    # State number of chlorophylls per protein complex.
    PS_table = pd.DataFrame({'chl_a': [148, 108, 24],
                             'chl_b': [12, 49, 18],
                             'chl_tot': [160, 157, 42]}, index=['PSI-LHCI', 'CSM', 'LHCII'])
    #PSI-LHCI from Pan 2018; CSM from Su 2014; LHCII from Laisk 2014 

    #import leaf absorbance and absorbance by photosynthetic pigments. 
    parameters = pd.read_excel('model parameters fvCBwd model/parameters.xlsx', header=2)
    # Inter- and extrapolates the missing values in the absorption by non-photosynthetic pigmens.  
    known_abs_pig = np.transpose(np.array(parameters[parameters['alpha_pigments'].notna()][['Wavelength(nm)', 'alpha_pigments']]))
    x_new = np.array(parameters['Wavelength(nm)'])
    parameters['alpha_pigments'] = np.interp(x_new, known_abs_pig[0], known_abs_pig[1])
    absorbance_photopigments = np.array(parameters.alpha_pigments)
    
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
    
    #loading data
    #initialise light and simulation time
    section_minutes_Taylor = 65
    section_time_Taylor = section_minutes_Taylor * 60
    time_steps_model_Taylor = 60
    model_length_Taylor =int( 5 * section_time_Taylor / time_steps_model_Taylor)+1
    Ci=350*10e-6
    Ci_model = np.repeat([Ci], model_length_Taylor)
    Time_Taylor = np.arange(-1 * section_time_Taylor/60,1+4 * section_time_Taylor/60, time_steps_model_Taylor/60)
    
    t_plot_Taylor = (0, 4*section_time_Taylor/60)
    growth_conditions = ['Sun', 'shade']
    

    shade_literature = pd.read_csv('model parameters FvCBwd model/SHADE_adapted.csv', sep=';', decimal=',', names=['Time_min', 'Assimilation'])    
    shade_literature.iat[0,0] = 0
    sun_literature = pd.read_csv('model parameters FvCBwd model/SUN_adapted.csv', sep=';', decimal=',', names=['Time_min', 'Assimilation']) 
    sun_literature.iat[0,0] = 0
    sun_literature.iat[1,0] = 1

    output_folder = find_or_create_results_folder('Taylor')
        

    global Time_model
    Time_model = Time_Taylor
    index_Fm = np.where(Time_model==130)
    time_steps_model = time_steps_model_Taylor
        
    #selecting relevant data for comparing outputs
    #Load data
    global Assimilation_sun
    global Assimilation_shade
    Assimilation_sun_obs = np.array(sun_literature.Assimilation*1e-6)
    Assimilation_shade_obs = np.array(shade_literature.Assimilation*1e-6)
    literature_assimilation = [Assimilation_sun_obs, Assimilation_shade_obs]
    global Time
    Time_obs = np.array(sun_literature.Time_min)

    
    #Obtain a mask for extracting observed time points from modelled time points. 
    obs_mask = [time_step in Time_obs for time_step in Time_model]
        
    shade_Fm_literature = pd.read_csv('model parameters FvCBwd model/Fm__Fm_shade.csv', sep=',', decimal='.', names=['Time_min', 'Fm'])    
    sun_Fm_literature = pd.read_csv('model parameters FvCBwd model/Fm__Fm sun.csv', sep=',', decimal='.', names=['Time_min', 'Fm'])    
    literature_Fm_frames = [sun_Fm_literature, shade_Fm_literature]
    
    shade_qP_literature = pd.read_csv('model parameters FvCBwd model/qP_shade.csv', sep=',', decimal='.', names=['Time_min', 'qP'])    
    sun_qP_literature = pd.read_csv('model parameters FvCBwd model/qP_sun.csv', sep=',', decimal='.', names=['Time_min', 'qP'])    
    literature_qP = [sun_qP_literature, shade_qP_literature]

    shade_qL_literature = pd.read_csv('model parameters FvCBwd model/qL_SHADE.csv', sep=';', decimal='.', names=['Time_min', 'qL'])    
    sun_qL_literature = pd.read_csv('model parameters FvCBwd model/qL_SUN.csv', sep=';', decimal='.', names=['Time_min', 'qL'])    
    literature_qL = [sun_qL_literature, shade_qL_literature]


    #reapeting for qT
    Fm_observed=[None,None]
    Fm_obs_masks = [None, None]
    Times_Fm = [None,None]
    Fm_interpolated_all = [None,None]
    Fm_observed_interpolated = [None,None]
    literature_Fm = [None,None]
    for i in [0,1]:
        
        Time_Fm = np.array(literature_Fm_frames[i].Time_min)
        Times_Fm[i] = Time_Fm
                
        Fm_obs_mask = np.zeros(len(Time_model), dtype=bool)
        for t in Time_Fm:
            idx = np.argmin(np.abs(Time_model - t))
            Fm_obs_mask[idx] = True

        Fm_observed[i] = np.array(literature_Fm_frames[i].Fm, dtype=float)
        Fm_obs_masks[i] = Fm_obs_mask
        interpolate = False

        literature_Fm[i] = np.array(literature_Fm_frames[i].Fm, dtype=float)


    #Obtaining light information and defining input light spectra
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
    spectrum_700_sun = load_spectrum('700_light', 55.6) #Correction factor is applied to get 40 umol absorbed photons        
    spectrum_700_shade = load_spectrum('700_light', 53.9) #Correction factor is applied to get 40 umol absorbed photons        

        
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

    light_lists = [light_list_Taylor_sun, light_list_Taylor_shade]
        
    leaf_absorbance_measured = np.array(parameters.leaf_absorbance)

    #measured spectrum of the Dual-PAM fluorescence measuring light 
    PAM_spect = np.loadtxt('Dual_pam_measuring_light_spectrum.txt', delimiter='\t', skiprows=1)[:371,1]
    PAM_spect[:50] = 0
    
    epsilon_F1 = 2
    epsilon_F2 = 1
    
    
    K_F2 = 0.05 # ns-1, rate of fluorescence
    global K_U2
    K_U2 = 2.1 # ns-1
    K_D2 = 0.55 # ns-1
    
    K_D1 = 0.55 # ns-1, rate of constitutive heat loss
    K_F1 = 0.05 # ns-1, rate of fluorescence
    
    global prior_table
    prior_table = pd.read_excel('priors_of_uncertainty.xlsx')
    prior_table.index = prior_table.model_parameter
    
    timeseries_length = len(Time_model)
    steps_coord = range(timeseries_length )
    coords = {"lags": range(-1, 0),
              "steps": steps_coord, 
              "timeseries_length": range(timeseries_length)}
    perform_fitting=True
    if perform_fitting:
        global basic_model
        basic_model = pm.Model(coords=coords, check_bounds=False)
        
        global idata
        idata, basic_model = run_model()
        idata_sel = idata

        with basic_model:
            #creates model fit param with LOO
            log_likelihood = pm.compute_log_likelihood(idata_sel)
            #plant_test.loo = az.loo(idata_sel)
            
            #makes a posterior predictive sample
            pm.sample_posterior_predictive(idata_sel, extend_inferencedata=True, random_seed=rng)
            # creates a posterior predictive plot
            az.plot_ppc(idata_sel, num_pp_samples=100);
            
                    
            fig, ax = plt.subplots(1,1, figsize=(6,5))
            az.plot_hdi(Time_obs, idata_sel.posterior_predictive['Y_obs_sun'])
            ax.scatter(Time_obs, idata_sel.observed_data['Y_obs_sun'])
            
            az.plot_hdi(Time_obs, idata_sel.posterior_predictive['Y_obs_shade'])
            ax.scatter(Time_obs, idata_sel.observed_data['Y_obs_shade'])
            ax.set_xlabel('Time')
            ax.set_ylabel('assimilation')
            
            fig.tight_layout()
            save_name = os.path.join(output_folder,'A_data_for_fit.png')
            fig.savefig(save_name,dpi=400)
            plt.close()    
        
        #TODO
        #with open(f"output_fits/{plant_test.plantID}/fitted_pickle.pkl", 'wb') as file_location:
        #    pkl.dump(plant_test, file_location)

#%% Run all posterior analysis and plot results

        #plots all the posterior distributions with chains
        # Get all variable names
        fitted_vars = list(idata.posterior.data_vars)
        
        # Exclude the last 8 variables
        vars_to_plot = fitted_vars[:-14]
        
        fig, axes = plt.subplots(10,4, figsize=(24,20))
        az.plot_trace(idata,combined=True, axes=axes[:,0:2], var_names=vars_to_plot[:10], figsize = (24, len(vars_to_plot)))
        az.plot_trace(idata,combined=True, axes=axes[:,2:4], var_names=vars_to_plot[10:20], figsize = (24, len(vars_to_plot)))
        save_name = os.path.join(output_folder,'chains_all.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=400)
        plt.close()
        
        fig, axes = plt.subplots(10,4, figsize=(24,20))
        az.plot_trace(idata_sel,combined=True, axes=axes[:,0:2], var_names=vars_to_plot[:10], figsize = (24, len(vars_to_plot)))
        az.plot_trace(idata_sel,combined=True, axes=axes[:,2:4], var_names=vars_to_plot[10:20], figsize = (24, len(vars_to_plot)))
        save_name = os.path.join(output_folder,'chains_selected.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=400)
        plt.close()
        
        #az.plot_trace(idata_sel,combined=True, var_names=vars_to_plot[20:], figsize = (12, 3*len(vars_to_plot[20:])))
        save_name = os.path.join(output_folder,'chains_selected_remaining.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=400)
        plt.close()
        #prints a summary of the resulting posterior distributions
        
        summary_all = az.summary(idata, var_names=vars_to_plot, round_to=2)
        save_name = os.path.join(output_folder,'summary_all.csv')
        summary_all.to_csv(save_name)
        
        summary_sel = az.summary(idata_sel, var_names=vars_to_plot, round_to=2)
        save_name = os.path.join(output_folder,'summary_selected.csv')
        summary_sel.to_csv(save_name)
        
        #creates an overview plot of the fit with the data. 
        post = idata_sel.posterior

    #%%
        array_length = len(idata_sel.posterior["P680_excitation_sun"].quantile(0.5, dim=["draw", "chain"]).values*1e6)
        percentiles = [0.025,0.5,0.975]
        
        P680_excitation_sun = np.zeros((3,array_length))
        P680_excitation_shade = np.zeros((3,array_length))
        
        P700_excitation_sun = np.zeros((3,array_length))
        P700_excitation_shade = np.zeros((3,array_length))
        
        PQH2_sun = np.zeros((3,array_length))
        PQH2_shade = np.zeros((3,array_length))
        
        D_P680_sun = np.zeros((3,array_length))
        D_P680_shade = np.zeros((3,array_length))
        
        D_P700_sun = np.zeros((3,array_length))
        D_P700_shade = np.zeros((3,array_length))
        
        J_P680_forward_sun = np.zeros((3,array_length))
        J_P680_forward_shade = np.zeros((3,array_length))
        
        J_P700_forward_sun = np.zeros((3,array_length))
        J_P700_forward_shade = np.zeros((3,array_length))
        
        LHCII_per_PSII_sun = np.zeros((3,array_length))
        LHCII_per_PSII_shade = np.zeros((3,array_length))
        
        LHCII_per_PSI_sun = np.zeros((3,array_length))
        LHCII_per_PSI_shade = np.zeros((3,array_length))
        
        A_sun = np.zeros((3,array_length))
        A_shade = np.zeros((3,array_length))
        
        Fm_raw_sun = np.zeros((3,array_length))
        Fm_raw_shade = np.zeros((3,array_length))
        
        Fm_sun = np.zeros((3,array_length))
        Fm_shade = np.zeros((3,array_length))
        
        Fo_sun = np.zeros((3,array_length))
        Fo_shade = np.zeros((3,array_length))
        
        Fs_sun = np.zeros((3,array_length))
        Fs_shade = np.zeros((3,array_length))
        
        for i in range(3):
            P680_excitation_sun[i] = idata_sel.posterior["P680_excitation_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6
            P680_excitation_shade[i] = idata_sel.posterior["P680_excitation_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6

            P700_excitation_sun[i] = idata_sel.posterior["P700_excitation_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6
            P700_excitation_shade[i] = idata_sel.posterior["P700_excitation_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6

            PQH2_sun[i] = idata_sel.posterior["PQH2_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            PQH2_shade[i] = idata_sel.posterior["PQH2_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            D_P680_sun[i] = idata_sel.posterior["D_P680_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            D_P680_shade[i] = idata_sel.posterior["D_P680_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            D_P700_sun[i] = idata_sel.posterior["D_P700_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            D_P700_shade[i] = idata_sel.posterior["D_P700_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            J_P680_forward_sun[i] = idata_sel.posterior["J_P680_forward_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6
            J_P680_forward_shade[i] = idata_sel.posterior["J_P680_forward_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6

            J_P700_forward_sun[i] = idata_sel.posterior["J_P700_forward_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6
            J_P700_forward_shade[i] = idata_sel.posterior["J_P700_forward_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6

            LHCII_per_PSII_sun[i] = idata_sel.posterior["LHCII_per_PSII_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            LHCII_per_PSII_shade[i] = idata_sel.posterior["LHCII_per_PSII_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            LHCII_per_PSI_sun[i] = idata_sel.posterior["LHCII_per_PSI_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            LHCII_per_PSI_shade[i] = idata_sel.posterior["LHCII_per_PSI_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            A_sun[i] = idata_sel.posterior["A_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6
            A_shade[i] = idata_sel.posterior["A_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values*1e6

            Fm_raw_sun[i] = idata_sel.posterior["Fm_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            Fm_raw_shade[i] = idata_sel.posterior["Fm_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            Fm_sun[i] = Fm_raw_sun[i] / Fm_raw_sun[i][index_Fm]
            Fm_shade[i] = Fm_raw_shade[i] / Fm_raw_shade[i][index_Fm]
            
            Fo_sun[i] = idata_sel.posterior["Fo_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            Fo_shade[i] = idata_sel.posterior["Fo_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values

            Fs_sun[i] = idata_sel.posterior["Fs_sun"].quantile(percentiles[i], dim=["draw", "chain"]).values
            Fs_shade[i] = idata_sel.posterior["Fs_shade"].quantile(percentiles[i], dim=["draw", "chain"]).values
            
        
        qL_sun = ((Fm_raw_sun - Fs_sun) / (Fm_raw_sun - Fo_sun)) * (Fo_sun / Fs_sun)
        qL_shade = ((Fm_raw_shade - Fs_shade) / (Fm_raw_shade - Fo_shade)) * (Fo_shade / Fs_shade)

        light_changes = [0,65,130,195,260]
        texts = ['Blue LED', 'FR LED','Blue LED', 'FR LED']
        colors = ['lightskyblue', 'lightcoral','lightskyblue', 'lightcoral']
        max_t = light_changes[4]
                
    #%%

        line_b = Line2D([], [], color="blue")
        band_b = Patch(facecolor="blue", alpha=0.2)
        line_o = Line2D([], [], color="orange")
        band_o = Patch(facecolor="orange", alpha=0.2)
        
        fig, axs = plt.subplots(2,3, figsize=(12,8))

        axs[0,0].fill_between(Time_model, P680_excitation_sun[0], P680_excitation_sun[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,0].plot(Time_model, P680_excitation_sun[1], color='blue', label="P680_excitation")
        axs[0,0].fill_between(Time_model, P700_excitation_sun[0], P700_excitation_sun[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,0].plot(Time_model, P700_excitation_sun[1], color='orange', label="P700_excitation")
        
        axs[0,0].fill_between(Time_model, P680_excitation_shade[0], P680_excitation_shade[2], color='blue', alpha=0.2, label='95% credibility interval', ls='--')
        axs[0,0].plot(Time_model, P680_excitation_shade[1], color='blue', label="P680_excitation", ls='--')
        axs[0,0].fill_between(Time_model, P700_excitation_shade[0], P700_excitation_shade[2], color='orange', alpha=0.2, label='95% credibility interval', ls='--')
        axs[0,0].plot(Time_model, P700_excitation_shade[1], color='orange', label="P700_excitation", ls='--')
        
        axs[0,0].set_ylabel(r'Photosystem excitation $[\mu mol\ m^{-2} \ s^{-1}]$')
        axs[0,0].set_ylim(0, P700_excitation_sun[2].max()+5)
        axs[0,0].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))


        axs[0,1].fill_between(Time_model, PQH2_sun[0], PQH2_sun[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,1].plot(Time_model, PQH2_sun[1], color='blue', label="median PQH2")
        
        axs[0,1].fill_between(Time_model, PQH2_shade[0], PQH2_shade[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,1].plot(Time_model, PQH2_shade[1], color='blue', label="median PQH2", ls='--')
        
        axs[0,1].set_ylabel(r'$PQH_2/PQ_{tot}$')
        axs[0,1].set_ylim(0, PQH2_sun[2].max()+0.03)
        axs[0,1].legend(
            [(line_b, band_b)],
            [r'$PQH_2/PQ_{tot}$ (95% CI)'],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))
        
        
        axs[0,2].fill_between(Time_model, D_P680_sun[0], D_P680_sun[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P680_sun[1], color='blue', label="median D_P680")
        axs[0,2].fill_between(Time_model, D_P700_sun[0], D_P700_sun[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P700_sun[1], color='orange', label="median D_P700")
        
        axs[0,2].fill_between(Time_model, D_P680_shade[0], D_P680_shade[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P680_shade[1], color='blue', label="median D_P680", ls='--')
        axs[0,2].fill_between(Time_model, D_P700_shade[0], D_P700_shade[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P700_shade[1], color='orange', label="median D_P700", ls='--')
        
        axs[0,2].set_ylabel('Fraction open reaction centres')
        axs[0,2].set_ylim(0,1.1)
        axs[0,2].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['P680 (95% CI)', "P700 (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))
        
        
        axs[1,0].fill_between(Time_model, J_P680_forward_sun[0], J_P680_forward_sun[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P680_forward_sun[1], color='blue', label="electron flow D_P680")
        axs[1,0].fill_between(Time_model, J_P700_forward_sun[0], J_P700_forward_sun[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P700_forward_sun[1], color='orange', label="electron flow D_P700")
        
        axs[1,0].fill_between(Time_model, J_P680_forward_shade[0], J_P680_forward_shade[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P680_forward_shade[1], color='blue', label="electron flow D_P680", ls='--')
        axs[1,0].fill_between(Time_model, J_P700_forward_shade[0], J_P700_forward_shade[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P700_forward_shade[1], color='orange', label="electron flow D_P700", ls='--')
    
        axs[1,0].set_ylabel(r'Electron flow rate $[\mu mol\ m^{-2} \ s^{-1}]$')
        axs[1,0].set_ylim(0,J_P700_forward_sun[2].max()+5)
        axs[1,0].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['P680 (95% CI)', "P700 (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))

        
        axs[1,1].fill_between(Time_model, LHCII_per_PSII_sun[0], LHCII_per_PSII_sun[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSII_sun[1], color='blue', label="median LHCII_per_PSII")
        axs[1,1].fill_between(Time_model, LHCII_per_PSI_sun[0], LHCII_per_PSI_sun[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSI_sun[1], color='orange', label="median LHCII_per_PSI")
        
        axs[1,1].fill_between(Time_model, LHCII_per_PSII_shade[0], LHCII_per_PSII_shade[2], color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSII_shade[1], color='blue', label="median LHCII_per_PSII", ls='--')
        axs[1,1].fill_between(Time_model, LHCII_per_PSI_shade[0], LHCII_per_PSI_shade[2], color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSI_shade[1], color='orange', label="median LHCII_per_PSI", ls='--')
        
        
        axs[1,1].set_ylabel('LHCII per PS')
        axs[1,1].set_ylim(0,LHCII_per_PSII_sun[2].max()+0.4)
        axs[1,1].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))

        
        for i in range(2):
            for j in range(3):
                axs[i,j].set_xlim(0,max_t)
                axs[1,j].set_xlabel('Time (min)')
                for k in range(4):
                    
                    banner = mpatches.Rectangle((light_changes[k]/max_t, 0.94), (light_changes[k+1]-light_changes[k])/max_t, 0.06,
                                                transform=axs[i,j].transAxes,  # axes coords
                                                facecolor=colors[k],
                                                edgecolor=colors[k], linewidth=0.8)
                    axs[i,j].add_patch(banner)
                    axs[i,j].text( (light_changes[k+1]+light_changes[k])/(2*max_t), 0.975, texts[k], transform=axs[i,j].transAxes,
                            va="center_baseline", ha="center", fontsize=10, color="black")
                    axs[i,j].vlines(light_changes[k],-10,150, ls='--', color='lightgrey', zorder=0)

        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'thylakoid_intermediate.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 
        
#%%Overview plot 
        fig, axs= plt.subplots(1,3, figsize=(15,5))
        Sun_Shade_i = 0

        A_fig, = axs[0].plot(sun_literature.Time_min, literature_assimilation[Sun_Shade_i]*1e6, ls='', marker='o', color='blue', label='Measured')
        axs[0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        axs[0].set_ylim(literature_assimilation[Sun_Shade_i].max()*1e6-1,literature_assimilation[Sun_Shade_i].max()*1e6 + 0.15)
        axs[0].legend(loc='upper left', bbox_to_anchor=(0, 0.94))

        Fm_fig, = axs[1].plot(Times_Fm[Sun_Shade_i], 
                              literature_Fm[Sun_Shade_i] ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[1].set_ylabel('Fm (%)')
        axs[1].set_ylim(0.8,1.1)
        axs[1].legend(loc='upper left', bbox_to_anchor=(0, 0.94),)
        
        
        qL_fig, = axs[2].plot(literature_qL[Sun_Shade_i].Time_min, 
                              literature_qL[Sun_Shade_i].qL ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[2].set_ylabel('qL (%)')
        axs[2].set_ylim(0,1)
        axs[2].legend(loc='upper left', bbox_to_anchor=(0, 0.94),)
        
        for i in range(3):
            axs[i].set_xlabel('Time (min)')
            axs[i].set_xlim(0,max_t)
            for j in range(4):
                
                banner = mpatches.Rectangle((light_changes[j]/max_t, 0.94), (light_changes[j+1]-light_changes[j])/max_t, 0.06,
                                            transform=axs[i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[i].add_patch(banner)
                axs[i].text( (light_changes[j+1]+light_changes[j])/(2*max_t), 0.975, texts[j], transform=axs[i].transAxes,
                        va="center_baseline", ha="center", fontsize=12, color="black")
                axs[i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'A_qT_P700.png')
        plt.show()
        fig.savefig(save_name,dpi=400)
        plt.close() 

#%% with model
        fig, axs= plt.subplots(1,3, figsize=(15,5))
        

        axs[0].fill_between(Time_model, A_sun[0], A_sun[2], color='blue', alpha=0.2, label='95% model CI')
        axs[0].plot(Time_model, A_sun[1], color='blue', label="Modelled")
        
        axs[0].fill_between(Time_model, A_shade[0], A_shade[2], color='blue', alpha=0.2, label='95% model CI')
        axs[0].plot(Time_model, A_shade[1], color='blue', label="Modelled", ls='--')
        
        A_fig_sun, = axs[0].plot(sun_literature.Time_min, literature_assimilation[0]*1e6, ls='', marker='o', color='blue', label='Measured')
        A_fig_shade, = axs[0].plot(shade_literature.Time_min, literature_assimilation[1]*1e6, ls='', marker='o', color='blue', label='Measured')

        axs[0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        axs[0].set_ylim(literature_assimilation[0].max()*1e6-1,literature_assimilation[0].max()*1e6 + 0.15)
        
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[0].legend(
            [A_fig_sun, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', fontsize=9  )
                    
        
        axs[1].plot(Time_model, Fm_sun[1] , label='Modelled', color='blue')
        axs[1].fill_between(Time_model, Fm_sun[0], Fm_sun[2], color='blue', alpha=0.2, label='95% model CI')
        
        axs[1].plot(Time_model, Fm_shade[1] , label='Modelled', color='blue')
        axs[1].fill_between(Time_model, Fm_shade[0], Fm_shade[2], color='blue', alpha=0.2, label='95% model CI')
        
        Fm_fig_sun, = axs[1].plot(Times_Fm[0], literature_Fm[0] ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        Fm_fig_shade, = axs[1].plot(Times_Fm[1], literature_Fm[1] ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[1].set_ylabel('Fm (%)')
        axs[1].set_ylim(0.8,1.1)
        
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[1].legend(
            [Fm_fig_sun, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0.01, 0.94),
            fontsize=9         )
        
        axs[2].plot(Time_model, D_P680_sun[1] , label='Modelled', color='blue')
        axs[2].fill_between(Time_model, D_P680_sun[0], D_P680_sun[2], color='blue', alpha=0.2, label='95% model CI')
        
        axs[2].plot(Time_model, D_P680_shade[1] , label='Modelled', color='blue')
        axs[2].fill_between(Time_model, D_P680_shade[0],  D_P680_shade[2], color='blue', alpha=0.2, label='95% model CI')
        
        qL_fig_sun, = axs[2].plot(literature_qL[0].Time_min, 
                              literature_qL[0].qL ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        qL_fig_shade, = axs[2].plot(literature_qL[1].Time_min, 
                              literature_qL[1].qL ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        
        axs[2].set_ylabel('qL')
        axs[2].set_ylim(0,1)
        
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[2].legend(
            [qL_fig_sun, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0.01, 0.94),
            fontsize=9         )

        
        for i in range(3):
            axs[i].set_xlabel('Time (min)')
            axs[i].set_xlim(0,max_t)
            for j in range(4):
                
                banner = mpatches.Rectangle((light_changes[j]/max_t, 0.94), (light_changes[j+1]-light_changes[j])/max_t, 0.06,
                                            transform=axs[i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[i].add_patch(banner)
                axs[i].text( (light_changes[j+1]+light_changes[j])/(2*max_t), 0.975, texts[j], transform=axs[i].transAxes,
                        va="center_baseline", ha="center", fontsize=12, color="black")
                axs[i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'A_qT_P700_fit.png')
        fig.savefig(save_name,dpi=400)
        plt.show()
        plt.close() 

        #%% OVerview plot of the distributions
        def exp_transform(x):
            return np.exp(x)
        
        unit_list = [1e6, 1e6, 1e6, 1,1, 
                     1e6, 1,1,1,1,
                     1e3,1e6,1,1,1,
                     1e6,1,1,1e6,1,1,1]
        
        fig, axs = plt.subplots(5,5, figsize = (20,20))
        for i in range(5):
            for j in range(5):
                var_num = i*5+j
                if var_num>=len(vars_to_plot):
                    continue

                az.plot_posterior(idata_sel,var_names=vars_to_plot[var_num], ax = axs[i,j],
                                      transform=exp_transform, label='posterior')
                az.plot_kde(np.exp(prior[vars_to_plot[var_num]]).values.flatten(), ax=axs[i,j], 
                            label='prior', plot_kwargs={'linestyle':'--', 'c':'orange'})
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'post_prior_param_fit.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 
        
        fig, axs = plt.subplots(5,5, figsize = (20,20))
        for i in range(5):
            for j in range(5):
                var_num = i*5+j
                if var_num>=len(vars_to_plot):
                    continue
    
                az.plot_posterior(idata_sel,var_names=vars_to_plot[var_num], ax = axs[i,j],
                                      transform=exp_transform, label='posterior')

        fig.tight_layout()
        save_name = os.path.join(output_folder,'post_param_fit.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 

        #%%
        fig, ax = plt.subplots(1,1, figsize = (6,6))
        x, y, xerr, yerr = [], [], [[],[]], [[],[]]
        for i in range(len(vars_to_plot)):

            x_values = prior[vars_to_plot[i]].values.flatten()
            x.append(np.median(x_values))
            xerr[0].append(np.median(x_values)-np.percentile(x_values, 5))
            xerr[1].append(np.percentile(x_values, 95)-np.median(x_values))
            y_values = idata_sel.posterior[vars_to_plot[i]].values.flatten()
            y.append(np.median(y_values))
            yerr[0].append(np.median(y_values)-np.percentile(y_values, 5))
            yerr[1].append(np.percentile(y_values, 95)-np.median(y_values))
        
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls='', marker='o', markersize=4)
        limits = (-18,4)
        for i in range(len(vars_to_plot)):
            ax.annotate(vars_to_plot[i], (x[i]+1, y[i]-0.2))
        ax.plot(np.linspace(limits,100), np.linspace(limits,100), ls='-', marker='')
        
        ax.set_xlabel('log transformed prior')
        ax.set_ylabel('log transformed posterior')
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'diagonal_fit.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 

        #%% Makes overview plot for a single plant analysis with distributions
        sun_color = 'red' 
        shade_color = 'blue'
        fig = plt.figure(figsize=(15*1.3,11*1.3))
        gs = gridspec.GridSpec(11,15)
        axs = np.empty((3,5), dtype=object)
        
        axs[0,0] = fig.add_subplot(gs[0:5,0:5])
        axs[0,1] = fig.add_subplot(gs[0:5,5:10])
        axs[0,2] = fig.add_subplot(gs[0:5,10:15])
        
        for i in range(5):
            axs[1,i] = fig.add_subplot(gs[5:8,3*i:3*i+3])
            axs[2,i] = fig.add_subplot(gs[8:11,3*i:3*i+3])
        
        axs[0,0].fill_between(Time_model, A_sun[0], A_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        axs[0,0].plot(Time_model, A_sun[1], color=sun_color, label="Modelled")
        
        axs[0,0].fill_between(Time_model, A_shade[0], A_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        axs[0,0].plot(Time_model, A_shade[1], color=shade_color, label="Modelled", ls='--')
        
        A_fig_sun, = axs[0,0].plot(sun_literature.Time_min, literature_assimilation[0]*1e6, ls='', marker='o', color=sun_color, label='Measured')
        A_fig_shade, = axs[0,0].plot(shade_literature.Time_min, literature_assimilation[1]*1e6, ls='', marker='o', color=shade_color, label='Measured')
        axs[0,0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$', fontsize=15)
        axs[0,0].set_ylim(literature_assimilation[0].max()*1e6-1,literature_assimilation[0].max()*1e6 + 0.15)
        
        Sun_color = Patch(facecolor=sun_color)
        Shade_color = Patch(facecolor=shade_color)
        measured_dot = mlines.Line2D([], [], color='black', marker='o', ls='')
        line = Line2D([], [], color="black")
        band = Patch(facecolor="black", alpha=0.2)
        axs[0,0].legend(
            [Sun_color, Shade_color, measured_dot, (line, band)],
            ['Sun adapted', 'Shade adapted', 'Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', fontsize=15, ncols=2  )
                    

        axs[0,1].plot(Time_model, Fm_sun[1] , label='Modelled', color=sun_color)
        axs[0,1].fill_between(Time_model, Fm_sun[0], Fm_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        
        axs[0,1].plot(Time_model, Fm_shade[1] , label='Modelled', color=shade_color, ls='--')
        axs[0,1].fill_between(Time_model, Fm_shade[0], Fm_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        
        Fm_fig_sun, = axs[0,1].plot(Times_Fm[0], literature_Fm[0] ,
                              c=sun_color, label='Measured', marker='o', ls="", markersize=3)
        Fm_fig_shade, = axs[0,1].plot(Times_Fm[1], literature_Fm[1] ,
                              c=shade_color, label='Measured', marker='o', ls="", markersize=3)
        axs[0,1].set_ylabel('qT (%)', fontsize=15)
        axs[0,1].set_ylim(0.8,1.1)

        
        axs[0,2].plot(Time_model, D_P680_sun[1] , label='Modelled', color=sun_color)
        axs[0,2].fill_between(Time_model, D_P680_sun[0], D_P680_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        
        axs[0,2].plot(Time_model, D_P680_shade[1] , label='Modelled', color=shade_color)
        axs[0,2].fill_between(Time_model, D_P680_shade[0],  D_P680_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        
        
        qL_fig_sun, = axs[0,2].plot(literature_qL[0].Time_min, 
                              literature_qL[0].qL ,
                              c=sun_color, label='Measured', marker='o', ls="", markersize=3)
        qL_fig_shade, = axs[0,2].plot(literature_qL[1].Time_min, 
                              literature_qL[1].qL ,
                              c=shade_color, label='Measured', marker='o', ls="", markersize=3)
        
        axs[0,2].set_ylabel('qL')
        axs[0,2].set_ylim(0,1.2)
        


        
        for i in range(3):
            axs[0,i].set_xlabel('Time (min)', fontsize=15)
            axs[0,i].set_xlim(0,max_t)
            axs[0,i].tick_params(axis='both', which='major', labelsize=15)
            for j in range(0,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[0])/(max_t-light_changes[0]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[0]), 0.06,
                                            transform=axs[0,i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[0,i].add_patch(banner)
                axs[0,i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[0])/(2*(max_t-light_changes[0])), 0.975, texts[j], transform=axs[0,i].transAxes,
                        va="center_baseline", ha="center", fontsize=15, color="black")
                axs[0,i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
        #_____________________________________________________________________________
        def exp_transform_ST_rate(x):
            return np.exp(x)*60
        
        def exp_transform_milion(x):
            return np.exp(x)*1e6
        
        param_list = ['PSI_PSII_ratio_sun', 'PSI_PSII_ratio_shade','k_max_CB6F', 'forward_ST_rate', 
                      'reverse_ST_rate', 'eta_CET', 'eta_LET', 'K_P1', 'CB6F_per_chl', 'leaf_abs_change']
        factor_list = [1,1,1,1,1,1,1,1,1,1]
        xlabels = ['PSI/PSII ratio sun[-]','PSI/PSII ratio shade[-]',
                   'maximal CB6F rate $[mol\ e^-\ mol^{-1}\ CB6F \ s^{-1}]$',
                   r'$k_{1\rightarrow2}$ $[min^{-1}]$',r'$k_{2\rightarrow1}$ $[min^{-1}]$',
                   r'$\eta_{CET}$ [-]', r'$\eta_{LET}$ [-]', r'$K_{P1}$ $[ns^{-1}]$',
                   r'$C_{CB6F}$ $[mol\ CB6F\ mol^{-1}\ Chl]$', r'leaf absorptance change [-]']
        prior_list = [ 'PSI_PSII_ratio','PSI_PSII_ratio', 'k_max_CB6F', 'forward_ST_rate',
                      'reverse_ST_rate', 'eta_CET', 'eta_LET', 'K_p1', 'CB6F_per_chl', 'alpha leaf']

        
        for i in [1,2]:
            for j in [0,1,2,3,4]:
                var_num = (i-1)*5+j
                if var_num>=len(param_list):
                    continue

                if param_list[var_num]=='forward_ST_rate' or param_list[var_num]=='reverse_ST_rate':
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_ST_rate, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = solve_for_log_normal_parameters(0.005, 0.005)
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*60, pdf/60, label='prior', c='black', ls='--')
                    axs[i,j].set_xlim(0,0.5)
                    
                elif param_list[var_num]=='leaf_abs_change':
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = prior_table.at[prior_list[var_num], 'mean'], prior_table.at[prior_list[var_num], 'std']
                      
                    x = np.linspace(
                        norm.ppf(0.001, scale=params[1], loc=params[0]),
                        norm.ppf(0.999, scale=params[1], loc=params[0]),
                        500)
                    pdf = norm.pdf(x, scale=params[1], loc=params[0])
                    axs[i,j].plot(x, pdf, label='prior', c='black', ls='--')
                
                elif factor_list[var_num]==1:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = solve_for_log_normal_parameters(prior_table.at[prior_list[var_num], 'mean'], 
                                                             prior_table.at[prior_list[var_num], 'std'])
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x, pdf, label='prior', c='black', ls='--')
                    
                
                else:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_milion, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = solve_for_log_normal_parameters(prior_table.at[prior_list[var_num], 'mean']*1e-6, 
                                                             prior_table.at[prior_list[var_num], 'std']*1e-6)
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*1e6, pdf*1e-6, label='prior', c='black', ls='--')

                axs[i,j].set_title('')
                axs[i,j].legend([])
                for txt in list(axs[i,j].texts):
                    if "HDI" in txt.get_text():
                        txt.remove()
                axs[i,j].set_xlabel(xlabels[var_num], fontsize=15)
        axs[1,0].legend(fontsize=15)

        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'individual_plant_for_paper.png')
        fig.savefig(save_name,dpi=400)
        plt.show()
        plt.close() 
        
        
        #%% Makes overview plot for a single plant analysis for the paper
        sun_color = 'red' 
        shade_color = 'blue'
        fig, axs = plt.subplots(1,3, figsize=(12,4))

        
        axs[0].fill_between(Time_model, A_sun[0], A_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        axs[0].plot(Time_model, A_sun[1], color=sun_color, label="Modelled")
        
        axs[0].fill_between(Time_model, A_shade[0], A_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        axs[0].plot(Time_model, A_shade[1], color=shade_color, label="Modelled", ls='-')
        
        A_fig_sun, = axs[0].plot(sun_literature.Time_min, literature_assimilation[0]*1e6, ls='', 
                                 marker='o', color=sun_color, label='Measured', markersize=3)
        A_fig_shade, = axs[0].plot(shade_literature.Time_min, literature_assimilation[1]*1e6, ls='', 
                                   marker='o', color=shade_color, label='Measured', markersize=3)
        axs[0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        axs[0].set_ylim(literature_assimilation[0].max()*1e6-1,literature_assimilation[0].max()*1e6 + 0.15)
        
        Sun_color = Patch(facecolor=sun_color)
        Shade_color = Patch(facecolor=shade_color)
        measured_dot = mlines.Line2D([], [], color='black', marker='o', ls='')
        line = Line2D([], [], color="black")
        band = Patch(facecolor="black", alpha=0.2)
        axs[0].legend(
            [Sun_color, Shade_color, measured_dot, (line, band)],
            ['Sun', 'Shade', 'Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', ncols=2  )
                    

        axs[1].plot(Time_model, Fm_sun[1] , label='Modelled', color=sun_color)
        axs[1].fill_between(Time_model, Fm_sun[0], Fm_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        
        axs[1].plot(Time_model, Fm_shade[1] , label='Modelled', color=shade_color, ls='-')
        axs[1].fill_between(Time_model, Fm_shade[0], Fm_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        
        Fm_fig_sun, = axs[1].plot(Times_Fm[0], literature_Fm[0] ,
                              c=sun_color, label='Measured', marker='o', ls="", markersize=3)
        Fm_fig_shade, = axs[1].plot(Times_Fm[1], literature_Fm[1] ,
                              c=shade_color, label='Measured', marker='o', ls="", markersize=3)
        axs[1].set_ylabel('Fm\' (-)')
        axs[1].set_ylim(0.8,1.1)

        
        axs[2].plot(Time_model, D_P680_sun[1] , label='Modelled', color=sun_color)
        axs[2].fill_between(Time_model, D_P680_sun[0], D_P680_sun[2], color=sun_color, alpha=0.2, label='95% model CI')
        
        axs[2].plot(Time_model, D_P680_shade[1] , label='Modelled', color=shade_color)
        axs[2].fill_between(Time_model, D_P680_shade[0],  D_P680_shade[2], color=shade_color, alpha=0.2, label='95% model CI')
        
        
        qL_fig_sun, = axs[2].plot(literature_qL[0].Time_min, 
                              literature_qL[0].qL ,
                              c=sun_color, label='Measured', marker='o', ls="", markersize=3)
        qL_fig_shade, = axs[2].plot(literature_qL[1].Time_min, 
                              literature_qL[1].qL ,
                              c=shade_color, label='Measured', marker='o', ls="", markersize=3)
        
        axs[2].set_ylabel('qL')
        axs[2].set_ylim(0,1.2)
        


        
        for i in range(3):
            axs[i].set_xlabel('Time (min)')
            axs[i].set_xlim(0,max_t)
            axs[i].tick_params(axis='both', which='major')
            for j in range(0,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[0])/(max_t-light_changes[0]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[0]), 0.06,
                                            transform=axs[i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[i].add_patch(banner)
                axs[i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[0])/(2*(max_t-light_changes[0])), 0.975, texts[j], transform=axs[i].transAxes,
                        va="center_baseline", ha="center", fontsize=10, color="black")
                axs[i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
         
        fig.tight_layout()
        save_name = os.path.join(output_folder,'individual_plant_for_paper.png')
        fig.savefig(save_name,dpi=400)
        plt.show()
        plt.close() 
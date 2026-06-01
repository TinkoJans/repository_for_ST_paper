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

    return new_state
    #%%Loading st data
def solve_for_log_normal_parameters(mean, sigma):
    variance = np.log(sigma**2/mean**2 + 1)
    mu = np.log(mean) - variance/2
    return (mu, variance**0.5)

def run_model(plant):
    
    with basic_model:
        # Priors for unknown model parameters
        Vcmax_params = solve_for_log_normal_parameters(plant.prior_table.at['Vcmax', 'prior'][0]*1e-6, 
                                                       plant.prior_table.at['Vcmax', 'prior'][1]*1e-6)
        Vcmax = pm.Normal('Vcmax', mu=Vcmax_params[0], sigma=Vcmax_params[1])
        
        #R_day
        Rd_params = solve_for_log_normal_parameters(plant.prior_table.at['R_d', 'prior'][0]*1e-6, 
                                                      plant.prior_table.at['R_d', 'prior'][1]*1e-6)
        Rd = pm.Normal('Rd', mu=Rd_params[0], sigma=Rd_params[1])
        
        #Chl_per_area
        Chl_per_area_params = solve_for_log_normal_parameters(plant.prior_table.at['Chl_per_area', 'prior'][0]*1e-6, 
                                                      plant.prior_table.at['Chl_per_area', 'prior'][1]*1e-6)
        Chl_per_area = pm.Normal('Chl_per_area', mu=Chl_per_area_params[0], sigma=Chl_per_area_params[1])
        
        #eta_LET
        eta_LET_params = solve_for_log_normal_parameters(plant.prior_table.at['eta_LET', 'prior'][0], 
                                                       plant.prior_table.at['eta_LET', 'prior'][1])
        eta_LET = pm.Normal('eta_LET', mu=eta_LET_params[0], sigma=eta_LET_params[1])
        
        #eta_CET
        eta_CET_params = solve_for_log_normal_parameters(plant.prior_table.at['eta_CET', 'prior'][0], 
                                                       plant.prior_table.at['eta_CET', 'prior'][1])
        eta_CET = pm.Normal('eta_CET', mu=eta_CET_params[0], sigma=eta_CET_params[1])
        
        #gamma_star
        gamma_star_params = solve_for_log_normal_parameters(plant.prior_table.at['gamma_star', 'prior'][0]*1e-6, 
                                                       plant.prior_table.at['gamma_star', 'prior'][1]*1e-6)
        gamma_star = pm.Normal('gamma_star', mu=gamma_star_params[0], sigma=gamma_star_params[1])
        
        #K_P2           
        K_P2_params = solve_for_log_normal_parameters(plant.prior_table.at['K_p2', 'prior'][0], 
                                                       plant.prior_table.at['K_p2', 'prior'][1])
        K_P2 = pm.Normal('K_P2', mu=K_P2_params[0], sigma=K_P2_params[1])
        
        #K_P1
        K_P1_params = solve_for_log_normal_parameters(plant.prior_table.at['K_p1', 'prior'][0], 
                                                       plant.prior_table.at['K_p1', 'prior'][1])
        K_P1 = pm.Normal('K_P1', mu=K_P1_params[0], sigma=K_P1_params[1])
        
        #PS ratio
        PSI_PSII_ratio_params = solve_for_log_normal_parameters(plant.prior_table.at['PSI_PSII_ratio', 'prior'][0], 
                                                       plant.prior_table.at['PSI_PSII_ratio', 'prior'][1])
        PSI_PSII_ratio = pm.Normal('PSI_PSII_ratio', mu=PSI_PSII_ratio_params[0], sigma=PSI_PSII_ratio_params[1])
                
        #PQ_per_chl
        PQ_per_chl_params = solve_for_log_normal_parameters(plant.prior_table.at['PQ_per_chl', 'prior'][0], 
                                                      plant.prior_table.at['PQ_per_chl', 'prior'][1])
        PQ_per_chl = pm.Normal('PQ_per_chl', mu=PQ_per_chl_params[0], sigma=PQ_per_chl_params[1])
        
        #MM_constant_oxy
        MM_constant_oxy_params = solve_for_log_normal_parameters(plant.prior_table.at['MM_constant_oxy', 'prior'][0]*1e-3, 
                                                       plant.prior_table.at['MM_constant_oxy', 'prior'][1]*1e-3)
        MM_constant_oxy = pm.Normal('MM_constant_oxy', mu=MM_constant_oxy_params[0], sigma=MM_constant_oxy_params[1])
        
        #MM_constant_carboxy
        MM_constant_carboxy_params = solve_for_log_normal_parameters(plant.prior_table.at['MM_constant_carboxy', 'prior'][0]*1e-6, 
                                                       plant.prior_table.at['MM_constant_carboxy', 'prior'][1]*1e-6)
        MM_constant_carboxy = pm.Normal('MM_constant_carboxy', mu=MM_constant_carboxy_params[0], sigma=MM_constant_carboxy_params[1])
        
        #PS_per_chl
        PS_per_chl_params = solve_for_log_normal_parameters(plant.prior_table.at['PS_per_chl', 'prior'][0], 
                                                      plant.prior_table.at['PS_per_chl', 'prior'][1])
        PS_per_chl = pm.Normal('PS_per_chl', mu=PS_per_chl_params[0], sigma=PS_per_chl_params[1])
        
        #K_N2
        K_N2_params = solve_for_log_normal_parameters(plant.prior_table.at['K_N2', 'prior'][0], 
                                                      plant.prior_table.at['K_N2', 'prior'][1])
        K_N2 = pm.Normal('K_N2', mu=K_N2_params[0], sigma=K_N2_params[1])
        
        #K_D2
        K_D2_params = solve_for_log_normal_parameters(plant.prior_table.at['K_D2', 'prior'][0], 
                                                      plant.prior_table.at['K_D2', 'prior'][1])
        K_D2 = pm.Normal('K_D2', mu=K_D2_params[0], sigma=K_D2_params[1])
        #K_D2 = np.log(plant.prior_table.at['K_D2', 'prior'][0])
        
        #alpha leaf
        leaf_absorbance_change = pm.Normal('leaf_abs_change', mu=0, sigma=plant.prior_table.at['alpha leaf', 'prior'][1])
        
        #g_meso
        g_meso_params = solve_for_log_normal_parameters(plant.prior_table.at['g_m', 'prior'][0], 
                                                       plant.prior_table.at['g_m', 'prior'][1])
        g_meso = pm.Normal('g_meso', mu=g_meso_params[0], sigma=g_meso_params[1])
        
        #CB6F_recovery_rate
        CB6F_recovery_rate_params = solve_for_log_normal_parameters(plant.prior_table.at['CB6F_recovery_rate', 'prior'][0]*1e-6, 
                                                       plant.prior_table.at['CB6F_recovery_rate', 'prior'][1]*1e-6)
        CB6F_recovery_rate = pm.Normal('CB6F_recovery_rate', mu=CB6F_recovery_rate_params[0], sigma=CB6F_recovery_rate_params[1])
        
        #Phosphorylation and dephosphorylation
        phos_params = solve_for_log_normal_parameters(0.005, 0.005)
        phosphorylation_rate_constant = pm.Normal(name='forward_ST_rate', mu=phos_params[0], sigma=phos_params[1])
        
        dephos_params = solve_for_log_normal_parameters(0.005, 0.005)
        dephosphorylation_rate_constant = pm.Normal(name='reverse_ST_rate', mu=dephos_params[0], sigma=dephos_params[1])
        
        #chl_a_b
        chl_ab_params = solve_for_log_normal_parameters(plant.prior_table.at['Chl_a_b', 'prior'][0], 
                                                       plant.prior_table.at['Chl_a_b', 'prior'][1])
        chl_ab = pm.Normal('chl_ab', mu=chl_ab_params[0], sigma=chl_ab_params[1])
        
        #remaining error
        sigma_params = solve_for_log_normal_parameters(0.1e-6, 0.1e-6)
        sigma = pm.Normal('sigma', mu=sigma_params[0], sigma=sigma_params[1])
        
        
        # sigma_qT_params = solve_for_log_normal_parameters(5, 3)
        # sigma_qT = pm.Normal('sigma_qT', mu=sigma_qT_params[0], sigma=sigma_qT_params[1])

        #defining ST model parameters
      
        rate_tot_P680_open = pt.exp(K_P2) + pt.exp(K_N2) + pt.exp(K_D2) + K_F2 + K_U2
        rate_tot_P680_closed = pt.exp(K_N2) + pt.exp(K_D2) + K_F2 + K_U2
        open_sharing_rate = K_U2 / rate_tot_P680_open
        closed_sharing_rate = K_U2 / rate_tot_P680_closed
        
        K_X1 = pt.exp(K_P1) # ns-1, rate of dissipation in closed P700, set equal to K_P1
        rate_tot_P700_open = pt.exp(K_P1) + K_D1 + K_F1
        rate_tot_P700_closed = K_X1 + K_D1 + K_F1
        
        #initialise thylakoid membrane
        concentration_PSII = pt.exp(PS_per_chl) * (1 - (pt.exp(PSI_PSII_ratio) / (1 + pt.exp(PSI_PSII_ratio)))) # mol/mol Chl
        concentration_PSI = pt.exp(PS_per_chl) - concentration_PSII # mol/mol Chl
        global D_P680_tot
        D_P680_tot = concentration_PSII * pt.exp(Chl_per_area) # mol m-2
        global D_P700_tot
        D_P700_tot = concentration_PSI * pt.exp(Chl_per_area) # mol m-2

        global PQ_pool
        PQ_pool = pt.exp(PQ_per_chl) * pt.exp(Chl_per_area) # mol m-2
        
        #calculate LHCII amount from PSI/PSII ratio and from chl_ab ratio
        a = (PS_table.at['PSI-LHCI', 'chl_a'] - pt.exp(chl_ab) * PS_table.at['PSI-LHCI', 'chl_b']) / (1 + (1/pt.exp(PSI_PSII_ratio)))
        b = (PS_table.at['CSM', 'chl_a'] - pt.exp(chl_ab) * PS_table.at['CSM', 'chl_b'])/ (1 + pt.exp(PSI_PSII_ratio))
        c = pt.exp(chl_ab) * PS_table.at['LHCII', 'chl_b'] - PS_table.at['LHCII', 'chl_a']
        extra_LHCII_per_PS = (a + b) / c
        
        leaf_absorbance = leaf_absorbance_measured + leaf_absorbance_change
        
        #initalise initial state
        A = 0
        chloroplastic_carbon_init = Ci_model[0]
        LHCII_per_PSII_init = 2 + extra_LHCII_per_PS #LHCII_per_PSII_init
        LHCII_per_PSI_init = extra_LHCII_per_PS #LHCII_per_PSI_init
        D_P680_init = 1 #D_P680 is initiated fully open
        D_P700_init = 1 #D_P680 is initiated fully open
        PQH2_init = 0
        Fm_init = 1 # placeholder for Fm, value is not used
        Fm_2_init = 1 # placeholder for Fm, value is not used
        P680_excitation_init = 40 # placeholder for excitation, value is not used
        P700_excitation_init = 40 # placeholder for excitation, value is not used
        J_P680_forward_init = 40 # placeholder for flow rate, value is not used
        J_P700_forward_init = 40 # placeholder for flow rate, value is not used

        initial_state = [
            pt.as_tensor_variable(A, dtype='float64'),
            pt.as_tensor_variable(chloroplastic_carbon_init, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSII_init, dtype='float64'),
            pt.as_tensor_variable(LHCII_per_PSI_init, dtype='float64'),
            pt.as_tensor_variable(D_P680_init, dtype='float64'),
            pt.as_tensor_variable(D_P700_init, dtype='float64'),
            pt.as_tensor_variable(PQH2_init, dtype='float64'),
            pt.as_tensor_variable(Fm_init, dtype='float64'),
            pt.as_tensor_variable(Fm_2_init, dtype='float64'),
            pt.as_tensor_variable(P680_excitation_init, dtype='float64'),
            pt.as_tensor_variable(P700_excitation_init, dtype='float64'),
            pt.as_tensor_variable(J_P680_forward_init, dtype='float64'),
            pt.as_tensor_variable(J_P700_forward_init, dtype='float64'),
        ]
        
        
        #make tensors out of input parameters
        Ci = pt.as_tensor_variable(Ci_model)
        light_spect = pt.as_tensor_variable(light_list)

        #revert to non transformed variables
        dephosphorylation_rate_constant = pt.exp(dephosphorylation_rate_constant)
        phosphorylation_rate_constant = pt.exp(phosphorylation_rate_constant)
        Vcmax = pt.exp(Vcmax)
        Rd = pt.exp(Rd)
        eta_LET = pt.exp(eta_LET)
        eta_CET = pt.exp(eta_CET)
        gamma_star = pt.exp(gamma_star)
        MM_constant_oxy = pt.exp(MM_constant_oxy)
        MM_constant_carboxy = pt.exp(MM_constant_carboxy)
        CB6F_recovery_rate = pt.exp(CB6F_recovery_rate)
        g_meso = pt.exp(g_meso)
        K_P2 = pt.exp(K_P2)
        K_P1 = pt.exp(K_P1)

            
        """This function loops over the complete timerange that is modelled. It takes
        an initial state and parameters needed for running the model. Ci and light_spect are different for each
        data point and are thus a list. THe other parameters are constant and are single values. """
        results, _ = pytensor.scan(
            fn = take_time_step,
            outputs_info=[{"initial": s, "taps": [-1]} for s in initial_state],
            sequences = [Ci[:], light_spect[:]],
            non_sequences=[dephosphorylation_rate_constant, phosphorylation_rate_constant,
                           Vcmax, Rd, eta_LET, eta_CET, gamma_star,
                           MM_constant_oxy, MM_constant_carboxy,
                           CB6F_recovery_rate, g_meso, K_P2, K_P1,
                           D_P680_tot, D_P700_tot, PQ_pool, 
                           rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
                           open_sharing_rate, closed_sharing_rate, leaf_absorbance,
                           time_steps_model],
            n_steps = timeseries_length,
            strict=True)
            
        #Defines which parameters will be provided as output.
        A = pm.Deterministic("A", results[0], dims=("timeseries_length",))
        Cc = pm.Deterministic("Cc", results[1], dims=("timeseries_length",))
        LHCII_per_PSII = pm.Deterministic("LHCII_per_PSII", results[2], dims=("timeseries_length",))
        LHCII_per_PSI = pm.Deterministic("LHCII_per_PSI", results[3], dims=("timeseries_length",))
        D_P680 = pm.Deterministic("D_P680", results[4], dims=("timeseries_length",))
        D_P700 = pm.Deterministic("D_P700", results[5], dims=("timeseries_length",))
        PQH2 = pm.Deterministic("PQH2", results[6], dims=("timeseries_length",))
        Fm = pm.Deterministic("Fm", results[7], dims=("timeseries_length",))
        Fm_2 = pm.Deterministic("Fm_2", results[8], dims=("timeseries_length",))
        P680_excitation = pm.Deterministic("P680_excitation", results[9], dims=("timeseries_length",))
        P700_excitation = pm.Deterministic("P700_excitation", results[10], dims=("timeseries_length",))
        J_P680_forward = pm.Deterministic("J_P680_forward", results[11], dims=("timeseries_length",))
        J_P700_forward = pm.Deterministic("J_P700_forward", results[12], dims=("timeseries_length",))
        
        qT = 100*((Fm[index_Fm] - Fm) / Fm[index_Fm])
        qT_saved = pm.Deterministic("qT_saved", qT, dims=("timeseries_length",))
        
        qT_mod = qT[np.array(qT_obs_mask)]
        # Determines subset of modelled A values that are also observed. 
        A_obs = A[np.array(obs_mask)]
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=A_obs, sigma=pt.exp(sigma), observed=Assimilation_obs)
        #qT_obs = pm.Normal('qT_obs', mu=qT_mod, sigma = pt.exp(sigma_qT), observed = qT_observed)

        #do some prior predictive checks. Does the model make sense with the selected priors?
        global idata
        idata_prior_pred = pm.sample_prior_predictive(draws=1000, random_seed=rng)
        global prior
        prior = idata_prior_pred.prior
        #plots prior predictive checks
        fig, ax = plt.subplots(1,2, figsize=(6,5))
        az.plot_hdi(Time_obs, idata_prior_pred.prior_predictive['Y_obs'], ax=ax[0])
        #az.plot_hdi(Time_qT_obs.repeat(n_repeats), idata.prior_predictive['qT_obs'], ax=ax[1])
        Y_obs_samples = idata_prior_pred.prior_predictive['Y_obs'].values
        #qT_obs_samples = idata.prior_predictive['qT_obs'].values
        
        fig, ax = plt.subplots(1,2, figsize=(6,5))
        #for i in range(qT_obs_samples.shape[1]):
        #    ax[0].plot(Time_qT_obs.repeat(n_repeats), qT_obs_samples[0,i,:], alpha=0.1, c='k')
        #ax[0].scatter(Time_qT_obs.repeat(n_repeats), idata.observed_data['qT_obs'])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('qT predicted')
        ax[0].set_title('Prior predictive checks')
        
        for i in range(Y_obs_samples.shape[1]):
           ax[1].plot(Time_obs, Y_obs_samples[0,i,:], alpha=0.1, c='k')
        ax[1].scatter(Time_obs, idata_prior_pred.observed_data['Y_obs'])
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('assimilation predicted')
        ax[1].set_title('Prior predictive checks')
        fig.tight_layout()
        save_name = os.path.join(output_folder,'prior_predictive.png')
        fig.savefig(save_name,dpi=400)
        plt.close() 

        global posterior
    
        n_chains = 3

        if False:
            with basic_model:
                #Run laplacian approximation
                idata_approx = pymc_extras.inference.fit_laplace()

        # Run sampling analysiss

        if False:
            n_chains = 12
            idata = pm.sample(cores=n_chains, chains=n_chains, tune=1000, draws=1000, 
                              progressbar=True,return_inferencedata=True,
                              init="jitter+adapt_diag", target_accept=0.9) 



        else:
            idata = idata_approx
        
        std_post = idata.posterior['Vcmax'].std(dim=['draw']).values
        good_chains = np.where(std_post>1e-4)[0]
        idata_sel = idata.isel(chain=good_chains)
        
        std_post_2 = idata_sel.posterior['MM_constant_oxy'].std(dim=['draw']).values
        good_chains_2 = np.where(std_post_2>0.05)[0]
        idata_sel = idata_sel.isel(chain=good_chains_2)

        std_post_3 = idata_sel.posterior['chl_ab'].std(dim=['draw']).values
        good_chains_3 = np.where(std_post_3>0.01)[0]
        idata_sel = idata_sel.isel(chain=good_chains_3)
        
        idata_posterior_pred = pm.sample_posterior_predictive(idata_sel).posterior_predictive
    return idata, basic_model, idata_prior_pred, idata_posterior_pred, idata_sel

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
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple


    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    base_directory = os.path.dirname(os.path.realpath(__file__))
    
    def find_or_create_results_folder(plantID):
        folder_name = 'output_fits'
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
    
    
    chloroplastic_oxygen = 212.7*1e-3 # bar, 21% of atmospheric pressure
        
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
    test_plants = ['SF3', 'SF4', 'SF5', 'SF6', 'SF7', 'SF8'] #'NT2', 'NT3', 'NT4', 'NT5', 'NT8', 'NT9',
                   #'Ph3', 'Ph4', 'Ph5', 'Ph6', 'Ph9',
                   #'SF3', 'SF4', 'SF5', 'SF6', 'SF7', 'SF8']
    
    for test_plant in test_plants:
        if os.path.isfile(f'./pickles_with_priors/{test_plant}.pkl'):
            print(f'loaded {test_plant} data from pickle')
            with open(f'pickles_with_priors/{test_plant}.pkl', 'rb') as file:
                plant_test = pkl.load(file)
                priors = plant_test.prior_table
        output_folder = find_or_create_results_folder(plant_test.plantID)
        
        #selecting relevant data for running model
        #Load data
        Ci_model = np.array(plant_test.gas_exchange_data.Ci)*1e-6
        global Time_model
        Time_model = np.array(plant_test.gas_exchange_data.minutes)
        
        #select relevant timepoints
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            time_points = np.where((Time_model>=0)&(Time_model<=88)) 
        else:
            time_points = np.where((Time_model>=0)&(Time_model<=78)) 
        Ci_model = Ci_model[time_points]
        Time_model = Time_model[time_points]
        
        #finds the index where Fm is defined
        global index_Fm
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            index_Fm = np.where((Time_model<=72)&(Time_model>71.91667))
        else:
            index_Fm = np.where((Time_model<=62)&(Time_model>61.91667))
        
        # define time step between modelling timepoints
        time_steps_model = 5
        
        #selecting relevant data for comparing outputs
        #Load data
        global Assimilation
        Assimilation = np.array(plant_test.gas_exchange_data_no_SP.A)*1e-6
        global Time
        Time = np.array(plant_test.gas_exchange_data_no_SP.minutes)

        
        #TODO
        #selecting relevant time points for Assimilation
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            time_points = np.where((Time>=40) &(Time<=88))
        else:
            time_points = np.where((Time>=30) &(Time<=78))
        Assimilation_obs = Assimilation[time_points]
        global Time_obs
        Time_obs = Time[time_points]
        
        #Obtain a mask for extracting observed time points from modelled time points. 
        obs_mask = [time_step in Time_obs for time_step in Time_model]
        
        #TODO
        #repeating for qT
        Time_qT = np.array(plant_test.dual_pam_data.minutes)
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            Time_qT_obs = Time_qT[np.where((Time_qT>=40) &(Time_qT<=88))]
        else:
            Time_qT_obs = Time_qT[np.where((Time_qT>=30) &(Time_qT<=78))]
            
        qT_obs_mask = np.zeros(len(Time_model), dtype=bool)
        for t in Time_qT_obs:
            idx = np.argmin(np.abs(Time_model - t))
            qT_obs_mask[idx] = True
        if plant_test.plantID[0:2] == 'SF':
            qT_observed_all = 100*(plant_test.dual_pam_data.at[21, 'Fm_prime'] - plant_test.dual_pam_data['Fm_prime']) / plant_test.dual_pam_data.at[21, 'Fm_prime'] 
        elif plant_test.plantID == 'NT8' or plant_test.plantID == 'NT9':
            qT_observed_all = 100*(plant_test.dual_pam_data.at[27, 'Fm_prime'] - plant_test.dual_pam_data['Fm_prime']) / plant_test.dual_pam_data.at[27, 'Fm_prime'] 
        elif plant_test.plantID[0:2] != 'SF':
            qT_observed_all = 100*(plant_test.dual_pam_data.at[25, 'Fm_prime'] - plant_test.dual_pam_data['Fm_prime']) / plant_test.dual_pam_data.at[25, 'Fm_prime']
        
        #TODO
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            qT_observed = np.array(qT_observed_all[np.where((Time_qT>=40) &(Time_qT<=88))[0]], dtype=float)
        else:
            qT_observed = np.array(qT_observed_all[np.where((Time_qT>=30) &(Time_qT<=78))[0]], dtype=float)

        #Obtaining light information and defining input light spectra
        WL_spect = plant_test.WL_spectrum * absorbance_photopigments
        WL_FR_spect = plant_test.WL_FR_spectrum * absorbance_photopigments
        light_list = np.zeros((len(Time_model),371))
        
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            for i in range(len(light_list)):
                if ((Time_model[i]>56) & (Time_model[i]<72)) or (Time_model[i]<40):
                    light_list[i] = WL_FR_spect*1e-6
                else:
                    light_list[i] = WL_spect*1e-6
        else:
            for i in range(len(light_list)):
                if ((Time_model[i]>46) & (Time_model[i]<62)) or (Time_model[i]<30):
                    light_list[i] = WL_FR_spect*1e-6
                else:
                    light_list[i] = WL_spect*1e-6
        
        leaf_absorbance_measured = plant_test.leaf_absorbance

        #measured spectrum of the Dual-PAM fluorescence measuring light 
        PAM_spect = np.loadtxt('Dual_pam_measuring_light_spectrum.txt', delimiter='\t', skiprows=1)[:371,1]
        PAM_spect[:50] = 0
        
        epsilon_F1 = plant_test.epsilon_PSI
        epsilon_F2 = plant_test.epsilon_PSII
        
        
        K_F2 = 0.05 # ns-1, rate of fluorescence
        global K_U2
        K_U2 = 2.1 # ns-1, to do
        
        K_D1 = 0.55 # ns-1, rate of constitutive heat loss
        K_F1 = 0.05 # ns-1, rate of fluorescence
        
        timeseries_length = len(Time_model)
        steps_coord = range(timeseries_length )
        coords = {"lags": range(-1, 0),
                  "steps": steps_coord, 
                  "timeseries_length": range(timeseries_length)}
        perform_fitting=False
        #TODO
        if perform_fitting:
            global basic_model
            basic_model = pm.Model(coords=coords, check_bounds=False)
            
            global idata
            idata, basic_model, idata_prior_pred, idata_posterior_pred, idata_sel = run_model(plant_test)
            plant_test.idata = idata

            plant_test.idata_prior_pred = idata_prior_pred
            plant_test.idata_posterior_pred = idata_posterior_pred

            idata_sel = idata_sel
            plant_test.idata_sel = idata_sel
            plant_test.prior = prior
            
            with basic_model:
                #creates model fit param with LOO
                plant_test.log_likelihood = pm.compute_log_likelihood(idata_sel)
                #plant_test.loo = az.loo(idata_sel)
                
                #makes a posterior predictive sample
                pm.sample_posterior_predictive(idata_sel, extend_inferencedata=True, random_seed=rng)
                # creates a posterior predictive plot
                az.plot_ppc(idata_sel, num_pp_samples=100);
                
                        
                fig, ax = plt.subplots(1,1, figsize=(6,5))
                az.plot_hdi(Time_obs, idata_sel.posterior_predictive['Y_obs'])
                ax.scatter(Time_obs, idata_sel.observed_data['Y_obs'])
                ax.set_xlabel('Time')
                ax.set_ylabel('assimilation')
                
                fig.tight_layout()
                save_name = os.path.join(output_folder,'A_data_for_fit.png')
                fig.savefig(save_name,dpi=400)
                plt.close()    
            
            
            with open(f"output_fits/{plant_test.plantID}/fitted_pickle.pkl", 'wb') as file_location:
                pkl.dump(plant_test, file_location)
                
        #TODO
        if not perform_fitting:
            #with open(f"output_fits_sampled/{plant_test.plantID}/fitted_pickle.pkl", 'rb') as file:
            with open(f"output_fits/{plant_test.plantID}/fitted_pickle.pkl", 'rb') as file:
                print(f'start loading {plant_test.plantID}')
                plant_object = pkl.load(file)
                print('finished loading')
                idata = plant_object.idata
                if True:
                    idata_prior_pred =plant_object.idata_prior_pred
                    idata_posterior_pred = plant_object.idata_posterior_pred
                
                #There is a problem that with some chains there chains seem to 
                #converge really quickly. THis leads to super narrow posteriors
                #with very bad fits. These are removed here. 
                # std_post = idata.posterior['Vcmax'].std(dim=['draw']).values
                # good_chains = np.where(std_post>1e-4)[0]
                # idata_sel = idata.isel(chain=good_chains)
                
                # std_post_2 = idata_sel.posterior['MM_constant_oxy'].std(dim=['draw']).values
                # good_chains_2 = np.where(std_post_2>0.05)[0]
                # idata_sel = idata_sel.isel(chain=good_chains_2)

                
                idata_sel = plant_object.idata_sel
                prior = plant_object.prior

#%% Run all posterior analysis and plot results


        #plots all the posterior distributions with chains
        # Get all variable names
        fitted_vars = list(idata.posterior.data_vars)
        
        # Exclude the last 13 variables
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
        
        az.plot_trace(idata_sel,combined=True, var_names=vars_to_plot[20:], figsize = (12, 3*len(vars_to_plot[20:])))
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
        P680_excitation_median = idata_sel.posterior["P680_excitation"].quantile(0.5, dim=["draw", "chain"]).values*1e6
        P680_excitation_low = idata_sel.posterior["P680_excitation"].quantile(0.025, dim=["draw", "chain"]).values*1e6
        P680_excitation_high = idata_sel.posterior["P680_excitation"].quantile(0.975, dim=["draw", "chain"]).values*1e6
        
        P700_excitation_median = idata_sel.posterior["P700_excitation"].quantile(0.5, dim=["draw", "chain"]).values*1e6
        P700_excitation_low = idata_sel.posterior["P700_excitation"].quantile(0.025, dim=["draw", "chain"]).values*1e6
        P700_excitation_high = idata_sel.posterior["P700_excitation"].quantile(0.975, dim=["draw", "chain"]).values*1e6
        
        PQH2_median = idata_sel.posterior["PQH2"].quantile(0.5, dim=["draw", "chain"]).values
        PQH2_low = idata_sel.posterior["PQH2"].quantile(0.025, dim=["draw", "chain"]).values
        PQH2_high = idata_sel.posterior["PQH2"].quantile(0.975, dim=["draw", "chain"]).values
       
        D_P680_median = idata_sel.posterior["D_P680"].quantile(0.5, dim=["draw", "chain"]).values
        D_P680_low = idata_sel.posterior["D_P680"].quantile(0.025, dim=["draw", "chain"]).values
        D_P680_high = idata_sel.posterior["D_P680"].quantile(0.975, dim=["draw", "chain"]).values
        
        D_P700_median = idata_sel.posterior["D_P700"].quantile(0.5, dim=["draw", "chain"]).values
        D_P700_low = idata_sel.posterior["D_P700"].quantile(0.025, dim=["draw", "chain"]).values
        D_P700_high = idata_sel.posterior["D_P700"].quantile(0.975, dim=["draw", "chain"]).values
        
        J_P680_forward_median = idata_sel.posterior["J_P680_forward"].quantile(0.5, dim=["draw", "chain"]).values*1e6
        J_P680_forward_low = idata_sel.posterior["J_P680_forward"].quantile(0.025, dim=["draw", "chain"]).values*1e6
        J_P680_forward_high = idata_sel.posterior["J_P680_forward"].quantile(0.975, dim=["draw", "chain"]).values*1e6
        
        J_P700_forward_median = idata_sel.posterior["J_P700_forward"].quantile(0.5, dim=["draw", "chain"]).values*1e6
        J_P700_forward_low = idata_sel.posterior["J_P700_forward"].quantile(0.025, dim=["draw", "chain"]).values*1e6
        J_P700_forward_high = idata_sel.posterior["J_P700_forward"].quantile(0.975, dim=["draw", "chain"]).values*1e6
        
        LHCII_per_PSII_median = idata_sel.posterior["LHCII_per_PSII"].quantile(0.5, dim=["draw", "chain"]).values
        LHCII_per_PSII_low = idata_sel.posterior["LHCII_per_PSII"].quantile(0.025, dim=["draw", "chain"]).values
        LHCII_per_PSII_high = idata_sel.posterior["LHCII_per_PSII"].quantile(0.975, dim=["draw", "chain"]).values
        
        LHCII_per_PSI_median = idata_sel.posterior["LHCII_per_PSI"].quantile(0.5, dim=["draw", "chain"]).values
        LHCII_per_PSI_low = idata_sel.posterior["LHCII_per_PSI"].quantile(0.025, dim=["draw", "chain"]).values
        LHCII_per_PSI_high = idata_sel.posterior["LHCII_per_PSI"].quantile(0.975, dim=["draw", "chain"]).values
        
        A_median = idata_sel.posterior["A"].quantile(0.5,dim=["draw", "chain"]).values *1e6
        A_low = idata_sel.posterior["A"].quantile(0.025, dim=["draw", "chain"]).values *1e6
        A_high = idata_sel.posterior["A"].quantile(0.975, dim=["draw", "chain"]).values *1e6
        
        Fm_median = idata_sel.posterior["Fm"].quantile(0.5, dim=["draw", "chain"]).values
        Fm_low = idata_sel.posterior["Fm"].quantile(0.025, dim=["draw", "chain"]).values
        Fm_high = idata_sel.posterior["Fm"].quantile(0.975, dim=["draw", "chain"]).values
        
        qT_med = 100*(Fm_median[index_Fm][0] - Fm_median) / Fm_median[index_Fm][0]
        qT_low = 100*(Fm_low[index_Fm][0] - Fm_low) / Fm_low[index_Fm][0]
        qT_high = 100*(Fm_high[index_Fm][0] - Fm_high) / Fm_high[index_Fm][0]
        
        if plant_test.plantID == 'NT8'or plant_test.plantID == 'NT9':
            light_changes = [0,40,56,72,88]
        else:
            light_changes = [0,30,46,62,78]
        texts = ['WL + FR', 'WL', 'WL + FR', 'WL']
        colors = ['lightcoral', 'lightgrey', 'lightcoral', 'lightgrey']
        max_t = light_changes[4]
        
        Assimilation = Assimilation[np.where(Time<=max_t)]
        Time = Time[np.where(Time<=max_t)]
        
    #%%

        line_b = Line2D([], [], color="blue")
        band_b = Patch(facecolor="blue", alpha=0.2)
        line_o = Line2D([], [], color="orange")
        band_o = Patch(facecolor="orange", alpha=0.2)
        
        fig, axs = plt.subplots(2,3, figsize=(12,8))

        axs[0,0].fill_between(Time_model, P680_excitation_low, P680_excitation_high, color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,0].plot(Time_model, P680_excitation_median, color='blue', label="P680_excitation")
        axs[0,0].fill_between(Time_model, P700_excitation_low, P700_excitation_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,0].plot(Time_model, P700_excitation_median, color='orange', label="P700_excitation")
        axs[0,0].set_ylabel(r'Photosystem excitation $[\mu mol\ m^{-2} \ s^{-1}]$')
        axs[0,0].set_ylim(0, P700_excitation_high.max()+5)
        axs[0,0].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))


        axs[0,1].fill_between(Time_model, PQH2_low, PQH2_high, color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,1].plot(Time_model, PQH2_median, color='blue', label="median PQH2")
        axs[0,1].set_ylabel(r'$PQH_2/PQ_{tot}$')
        axs[0,1].set_ylim(0, PQH2_high.max()+0.03)
        axs[0,1].legend(
            [(line_b, band_b)],
            [r'$PQH_2/PQ_{tot}$ (95% CI)'],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))
        
        
        axs[0,2].fill_between(Time_model, D_P680_low, D_P680_high, color='blue', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P680_median, color='blue', label="median D_P680")
        axs[0,2].fill_between(Time_model, D_P700_low, D_P700_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, D_P700_median, color='orange', label="median D_P700")
        axs[0,2].set_ylabel('Fraction open reaction centres')
        axs[0,2].set_ylim(0,1.1)
        axs[0,2].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['P680 (95% CI)', "P700 (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))
        
        
        axs[1,0].fill_between(Time_model, J_P680_forward_low, J_P680_forward_high, color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P680_forward_median, color='blue', label="electron flow D_P680")
        axs[1,0].fill_between(Time_model, J_P700_forward_low, J_P700_forward_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, J_P700_forward_median, color='orange', label="electron flow D_P700")
        axs[1,0].set_ylabel(r'Electron flow rate $[\mu mol\ m^{-2} \ s^{-1}]$')
        axs[1,0].set_ylim(0,J_P700_forward_high.max()+5)
        axs[1,0].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['P680 (95% CI)', "P700 (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))

        
        axs[1,1].fill_between(Time_model, LHCII_per_PSII_low, LHCII_per_PSII_high, color='blue', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSII_median, color='blue', label="median LHCII_per_PSII")
        axs[1,1].fill_between(Time_model, LHCII_per_PSI_low, LHCII_per_PSI_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, LHCII_per_PSI_median, color='orange', label="median LHCII_per_PSI")
        axs[1,1].set_ylabel('LHCII per PS')
        axs[1,1].set_ylim(0,LHCII_per_PSII_high.max()+0.4)
        axs[1,1].legend(
            [(line_b, band_b), (line_o, band_o)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01))

        
        for i in range(2):
            for j in range(3):
                axs[i,j].set_xlim(30,max_t)
                axs[1,j].set_xlabel('Time (min)')
                for k in range(1,4):
                    
                    banner = mpatches.Rectangle(((light_changes[k]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[k+1]-light_changes[k])/(max_t-light_changes[1]), 0.06,
                                                transform=axs[i,j].transAxes,  # axes coords
                                                facecolor=colors[k],
                                                edgecolor=colors[k], linewidth=0.8)
                    axs[i,j].add_patch(banner)
                    axs[i,j].text( ((light_changes[k+1]+light_changes[k])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[k], transform=axs[i,j].transAxes,
                            va="center_baseline", ha="center", color="black")
                    axs[i,j].vlines(light_changes[k],-10,150, ls='--', color='lightgrey', zorder=0)

        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'thylakoid_intermediate.png')
        plt.show()
        fig.savefig(save_name,dpi=400)
        plt.close() 
        
        
        
#%%Overview plot 
        fig, axs= plt.subplots(1,3, figsize=(12,4))
        
        A_fig, = axs[0].plot(Time, Assimilation*1e6, ls='', marker='o', color='blue', label='Measured',
                             markersize=4)
        axs[0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        axs[0].set_ylim(Assimilation.max()*1e6 - 1,Assimilation.max()*1e6 + 0.15)
        axs[0].legend(loc='upper left', bbox_to_anchor=(0, 0.94))
                    

        qT_fig, = axs[1].plot(Time_qT, 
                              qT_observed_all ,
                              c='blue', label='Measured', marker='o', ls="")
        axs[1].set_ylabel('qT (%)')
        axs[1].set_ylim(-1, qT_observed_all.max() +2)
        axs[1].legend(loc='upper left', bbox_to_anchor=(0, 0.94),)
        
     
        
        donor_side_limitation_uncorrected = (plant_test.dual_pam_data.Ps-plant_test.dual_pam_data.P0)/plant_test.Pm
        QY_uncorrected = (plant_test.dual_pam_data.Phigh-plant_test.dual_pam_data.Ps)/plant_test.Pm
        donor_side_limitation = donor_side_limitation_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        QY = QY_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        
        
        donor_fig, = axs[2].plot(plant_test.dual_pam_data.minutes, donor_side_limitation,
                    c='orange', label='Donor side', ls='', marker='o')

        QY_fig, = axs[2].plot(plant_test.dual_pam_data.minutes, QY,
                    c='black', label='Photochemistry', ls='', marker='o')
        axs[2].set_ylabel('Relative P700 yield')
        axs[2].set_ylim(-0.1,1.4)        
        axs[2].legend(loc='upper left', bbox_to_anchor=(0, 0.94),)
        
        
        for i in range(3):
            axs[i].set_xlabel('Time (min)')
            axs[i].set_xlim(30,max_t)
            for j in range(1,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[1]), 0.06,
                                            transform=axs[i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[i].add_patch(banner)
                axs[i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[j], transform=axs[i].transAxes,
                        va="center_baseline", ha="center", color="black")
                axs[i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'A_qT_P700.png')
        plt.show()
        fig.savefig(save_name,dpi=400)
        plt.close() 


#%% with model
        fig, axs= plt.subplots(1,3, figsize=(12,3))
        

        axs[0].fill_between(Time_model, A_low, A_high, color='blue', alpha=0.2, label='95% model CI')
        axs[0].plot(Time_model, A_median, color='blue', label="Modelled")
        A_fig, = axs[0].plot(Time, Assimilation*1e6, ls='', marker='o', color='blue', label='Measured')
        axs[0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        axs[0].set_ylim(Assimilation.max()*1e6 - 1,Assimilation.max()*1e6 + 0.15)
        
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[0].legend(
            [A_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', fontsize=9  )
                    

        axs[1].plot(Time_model, qT_med , label='Modelled', color='blue')
        axs[1].fill_between(Time_model, qT_low, qT_high, color='blue', alpha=0.2, label='95% model CI')
        qT_fig, = axs[1].plot(Time_qT, 
                              qT_observed_all ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[1].set_ylabel('qT (%)')
        axs[1].set_ylim(-1, qT_observed_all.max() +2)
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[1].legend(
            [qT_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0.01, 0.94),
            fontsize=9         )
        

        QY = D_P700_median
        QY_low = D_P700_low
        QY_high = D_P700_high
        axs[2].plot(Time_model,1- QY ,c='orange', ls='--', label='P700 donor limitation')
        axs[2].fill_between(Time_model, 1-QY_low, 1-QY_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[2].plot(Time_model, QY ,c='black',ls='--', label='P700 photochemistry')
        axs[2].fill_between(Time_model, QY_low, QY_high, color='black', alpha=0.2, label='95% credibility interval')

        
        donor_side_limitation_uncorrected = (plant_test.dual_pam_data.Ps-plant_test.dual_pam_data.P0)/plant_test.Pm
        QY_uncorrected = (plant_test.dual_pam_data.Phigh-plant_test.dual_pam_data.Ps)/plant_test.Pm
        donor_side_limitation = donor_side_limitation_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        QY = QY_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        
        
        donor_fig, = axs[2].plot(plant_test.dual_pam_data.minutes, donor_side_limitation,
                    c='orange', label='Donor side limitation', ls='', marker='o')

        QY_fig, = axs[2].plot(plant_test.dual_pam_data.minutes, QY,
                    c='black', label='Photochemistry', ls='', marker='o')
        axs[2].set_ylabel('Relative P700 yield')
        axs[2].set_ylim(-0.1,1.5)
        axs[2].legend(loc='upper left')
        
        modelled_line = Line2D([],[], c='black', ls='--' )
        modelled_band = Patch(facecolor='black', alpha=0.2)
        QY_patch = Patch(facecolor='black', alpha=1)
        donor_patch = Patch(facecolor='orange', alpha=1)


        axs[2].legend(
            [QY_fig, QY_patch, (modelled_line, modelled_band), donor_patch],
            [r'Measured', 'Photochemistry',
             r"Modelled (95% CI)", "Donor limitation"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0, 0.94),
            ncols=2
        )

        

        
        for i in range(3):
            axs[i].set_xlabel('Time (min)')
            axs[i].set_xlim(30,max_t)
            for j in range(1,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[1]), 0.06,
                                            transform=axs[i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[i].add_patch(banner)
                axs[i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[j], transform=axs[i].transAxes,
                        va="center_baseline", ha="center", color="black")
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
        def exp_transform_thousand(x):
            return np.exp(x) *1e3
        def exp_transform_milion(x):
            return np.exp(x)*1e6
        param_list = ['Rd', 'PSI_PSII_ratio', 'CB6F_recovery_rate', 'forward_ST_rate', 
                      'reverse_ST_rate', 'chl_ab']
        factor_list = [1e6,1,1e6,1,1,1]
        fig, axs = plt.subplots(3,3, figsize=(10,10))
        for i in [0,1,2]:
            for j in [0,1,2]:
                var_num = i*3+j
                if var_num>=len(param_list):
                    continue

                if factor_list[var_num]==1:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform, label='posterior')
                    az.plot_kde(np.exp(prior[param_list[var_num]]).values.flatten(), ax=axs[i,j], 
                                label='prior', plot_kwargs={'linestyle':'--', 'c':'orange'})
                else:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_milion, label='posterior')
                    az.plot_kde(np.exp(prior[param_list[var_num]]).values.flatten()*1e6, ax=axs[i,j], 
                                label='prior', plot_kwargs={'linestyle':'--', 'c':'orange'})
                #axs[i,j].legend()
        axs[1,0].set_xlim(0,np.exp(idata_sel.posterior.forward_ST_rate).max())
        axs[1,1].set_xlim(0,np.exp(idata_sel.posterior.reverse_ST_rate).max())
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'post_prior_selected_param_fit.png')
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
        
        
        
        #%% Makes overview plot for a single plant analysis for the paper
        fontsize = 19
        fig = plt.figure(figsize=(15*1.3,16*1.3))
        gs = gridspec.GridSpec(16,15)
        axs = np.empty((4,5), dtype=object)
        
        axs[0,0] = fig.add_subplot(gs[0:5,0:5])
        axs[0,1] = fig.add_subplot(gs[0:5,5:10])
        axs[0,2] = fig.add_subplot(gs[0:5,10:15])
        
        axs[1,0] = fig.add_subplot(gs[5:10,0:5])
        axs[1,1] = fig.add_subplot(gs[5:10,5:10])
        axs[1,2] = fig.add_subplot(gs[5:10,10:15])
        
        for i in range(5):
            axs[2,i] = fig.add_subplot(gs[10:13,3*i:3*i+3])
            axs[3,i] = fig.add_subplot(gs[13:16,3*i:3*i+3])
        
        axs[0,0].fill_between(Time_model, A_low, A_high, color='orange', alpha=0.2, label='95% model CI')
        axs[0,0].plot(Time_model, A_median, color='orange', label="Modelled")
        A_fig, = axs[0,0].plot(Time, Assimilation*1e6, ls='', marker='o', color='orange', label='Measured')
        axs[0,0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$', fontsize=fontsize)
        axs[0,0].set_ylim(Assimilation.max()*1e6 - 1,Assimilation.max()*1e6 + 0.15)
        
        line = Line2D([], [], color="orange")
        band = Patch(facecolor="orange", alpha=0.2)
        axs[0,0].legend(
            [A_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', fontsize=fontsize  )
                    

        axs[0,1].plot(Time_model, qT_med , label='Modelled', color='blue')
        axs[0,1].fill_between(Time_model, qT_low, qT_high, color='blue', alpha=0.2, label='95% model CI')
        qT_fig, = axs[0,1].plot(Time_qT, 
                              qT_observed_all ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[0,1].set_ylabel('qT (%)', fontsize=fontsize)
        axs[0,1].set_ylim(-1, qT_observed_all.max() +2)
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[0,1].legend(
            [qT_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0.01, 0.94),
            fontsize=fontsize         )
        

        QY = D_P700_median
        QY_low = D_P700_low
        QY_high = D_P700_high
        axs[0,2].plot(Time_model,1- QY ,c='orange', ls='--', label='P700 donor limitation')
        axs[0,2].fill_between(Time_model, 1-QY_low, 1-QY_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, QY ,c='black',ls='--', label='P700 photochemistry')
        axs[0,2].fill_between(Time_model, QY_low, QY_high, color='black', alpha=0.2, label='95% credibility interval')

        
        donor_side_limitation_uncorrected = (plant_test.dual_pam_data.Ps-plant_test.dual_pam_data.P0)/plant_test.Pm
        QY_uncorrected = (plant_test.dual_pam_data.Phigh-plant_test.dual_pam_data.Ps)/plant_test.Pm
        donor_side_limitation = donor_side_limitation_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        QY = QY_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        
        
        donor_fig, = axs[0,2].plot(plant_test.dual_pam_data.minutes, donor_side_limitation,
                    c='orange', label='Donor side limitation', ls='', marker='o')

        QY_fig, = axs[0,2].plot(plant_test.dual_pam_data.minutes, QY,
                    c='black', label='Photochemistry', ls='', marker='o')

        axs[0,2].set_ylabel(r'$\phi/(\phi_I + \phi_{ND})$', fontsize=fontsize)
        axs[0,2].set_ylim(-0.1,1.4)

        

        modelled_line = Line2D([],[], c='black', ls='--' )
        modelled_band = Patch(facecolor='black', alpha=0.2)
        QY_patch = Patch(facecolor='black', alpha=1)
        donor_patch = Patch(facecolor='orange', alpha=1)


        axs[0,2].legend(
            [QY_fig, QY_patch, (modelled_line, modelled_band), donor_patch],
            [r'Measured', 'Photochemistry',
             r"Modelled (95% CI)", "Donor limitation"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0, 0.94),
            ncols=2, fontsize=fontsize-2
        )

        
        for i in range(3):
            #axs[0,i].set_xlabel('Time (min)', fontsize=fontsize)
            axs[0,i].set_xlim(30,max_t)
            axs[0,i].tick_params(axis='both', which='major', labelsize=fontsize)
            for j in range(1,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[1]), 0.06,
                                            transform=axs[0,i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[0,i].add_patch(banner)
                axs[0,i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[j], transform=axs[0,i].transAxes,
                        va="center_baseline", ha="center", fontsize=18, color="black")
                axs[0,i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        #_______________________________________________________________________________
        
        
        
        line_b = Line2D([], [], color="blue")
        band_b = Patch(facecolor="blue", alpha=0.2)
        line_o = Line2D([], [], color="orange")
        band_o = Patch(facecolor="orange", alpha=0.2)
        line_black = Line2D([], [], color="black")
        band_black = Patch(facecolor="black", alpha=0.2)
        line_black_dashed = Line2D([], [], color="black", ls='--')
        band_black_dashed = Patch(facecolor="black", alpha=0.2)

        axs[1,0].fill_between(Time_model, P680_excitation_low, P680_excitation_high, color='black', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, P680_excitation_median, color='black', label="P680_excitation", ls='--')
        axs[1,0].fill_between(Time_model, P700_excitation_low, P700_excitation_high, color='black', alpha=0.2, label='95% credibility interval')
        axs[1,0].plot(Time_model, P700_excitation_median, color='black', label="P700_excitation")
        axs[1,0].set_ylabel(r'Photosystem excitation $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
        axs[1,0].set_ylim(0, P700_excitation_high.max()+5)
        axs[1,0].legend(
            [(line_black, band_black), (line_black_dashed, band_black_dashed)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize=fontsize)


        axs[1,1].fill_between(Time_model, PQH2_low, PQH2_high, color='orange', alpha=0.2, label='95% credibility interval')
        axs[1,1].plot(Time_model, PQH2_median, color='orange', label="median PQH2")
        axs[1,1].set_ylabel(r'$PQH_2/PQ_{tot}$', fontsize=fontsize)
        axs[1,1].set_ylim(0, PQH2_high.max()+0.03)
        axs[1,1].legend(
            [(line_o, band_o)],
            [r'$PQH_2/PQ_{tot}$ (95% CI)'],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize=fontsize)
        

        
        axs[1,2].fill_between(Time_model, LHCII_per_PSII_low, LHCII_per_PSII_high, color='black', alpha=0.2, label='95% credibility interval')
        axs[1,2].plot(Time_model, LHCII_per_PSII_median, color='black', label="median LHCII_per_PSII")
        axs[1,2].fill_between(Time_model, LHCII_per_PSI_low, LHCII_per_PSI_high, ls='--', color='black', alpha=0.2, label='95% credibility interval')
        axs[1,2].plot(Time_model, LHCII_per_PSI_median, color='black', ls='--', label="median LHCII_per_PSI")
        axs[1,2].set_ylabel('LHCII per PS', fontsize=fontsize)
        axs[1,2].set_ylim(0,LHCII_per_PSII_high.max()+0.4)
        axs[1,2].legend(
            [(line_black, band_black), (line_black_dashed, band_black_dashed)],
            ['PSII (95% CI)', "PSI (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower left', bbox_to_anchor=(0.01, 0.01),fontsize=fontsize)

        
        for i in range(1):
            for j in range(3):
                axs[1,j].set_xlim(30,max_t)
                axs[1,j].set_xlabel('Time (min)', fontsize=fontsize)
                axs[1,j].tick_params(axis='both', which='major', labelsize=fontsize)

                for k in range(1,4):
                    
                    banner = mpatches.Rectangle(((light_changes[k]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[k+1]-light_changes[k])/(max_t-light_changes[1]), 0.06,
                                                transform=axs[1,j].transAxes,  # axes coords
                                                facecolor=colors[k],
                                                edgecolor=colors[k], linewidth=0.8)
                    axs[1,j].add_patch(banner)
                    axs[1,j].text( ((light_changes[k+1]+light_changes[k])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[k], transform=axs[1,j].transAxes,
                            va="center_baseline", ha="center", color="black", fontsize=18)
                    axs[1,j].vlines(light_changes[k],-10,150, ls='--', color='lightgrey', zorder=0)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #_____________________________________________________________________________
        def exp_transform_ST_rate(x):
            return np.exp(x)*60
        
        def exp_transform_milion(x):
            return np.exp(x)*1e6
        
        def exp_transform_thousand(x):
            return np.exp(x)*1e3
        
        param_list = ['Rd', 'PSI_PSII_ratio', 'CB6F_recovery_rate', 'forward_ST_rate', 
                      'reverse_ST_rate', 'Chl_per_area', 'PS_per_chl', 'chl_ab',
                      'K_N2', 'g_meso']
        factor_list = [1e6,1,1e6,1,1,1e3, 1e3,1,1,1]
        xlabels = [r'$R_{day}$ $[\mu mol\ m^{-2}\ s^{-1}]$', 'PSI/PSII ratio [-]',
                   'CB6F recovery rate $[\mu mol\ m^{-2}\ s^{-1}]$',
                   r'$k_{1\rightarrow2}$ $[min^{-1}]$',r'$k_{2\rightarrow1}$ $[min^{-1}]$',
                   r'Chl content $[mmol\ m^{-2}]$', r'PS content $[mmol\ mol^{-1}]$',
                   'chl a/b ratio [-]', r'Rate of NPQ $[ns^{-1}]$', r'g_m [$mol\ m^{-2}\ s^{-1}\ bar^{-1}]$'
                   ]
        prior_list = ['R_d', 'PSI_PSII_ratio', 'CB6F_recovery_rate', 'forward_ST_rate',
                      'reverse_ST_rate', 'Chl_per_area', 'PS_per_chl', 'Chl_a_b',
                      'K_N2', 'g_m']

        
        for i in [2,3]:
            for j in [0,1,2,3,4]:
                var_num = (i-2)*5+j
                if var_num>=len(param_list):
                    continue

                if param_list[var_num]=='forward_ST_rate' or param_list[var_num]=='reverse_ST_rate':
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_ST_rate, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black', textsize=fontsize)

                    params = solve_for_log_normal_parameters(0.005, 0.005)
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*60, pdf/60, label='prior', c='black', ls='--')
                    
                    
                elif param_list[var_num]=='leaf_abs_change':
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black', textsize=fontsize)

                    params = priors.at[prior_list[var_num], 'prior'][0], priors.at[prior_list[var_num], 'prior'][1]
                      
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
                                          c='black', textsize=fontsize)

                    params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0], 
                                                             priors.at[prior_list[var_num], 'prior'][1])
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x, pdf, label='prior', c='black', ls='--')
                    
                elif factor_list[var_num]==1e3:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_thousand, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black', textsize=fontsize)
                    if param_list[var_num]=='Chl_per_area':
                        params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0]*1e-6, 
                                                                 priors.at[prior_list[var_num], 'prior'][1]*1e-6)
                    else:
                        params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0], 
                                                                 priors.at[prior_list[var_num], 'prior'][1])
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*1e3, pdf*1e-3, label='prior', c='black', ls='--')
                else:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_milion, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black', textsize=fontsize)

                    params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0]*1e-6, 
                                                             priors.at[prior_list[var_num], 'prior'][1]*1e-6)
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*1e6, pdf*1e-6, label='prior', c='black', ls='--')

                axs[i,j].set_title('')
                axs[i,j].legend([], fontsize=fontsize)
                for txt in list(axs[i,j].texts):
                    if "HDI" in txt.get_text():
                        txt.remove()
                axs[i,j].set_xlabel(xlabels[var_num], fontsize=fontsize)
        axs[2,0].legend(fontsize=fontsize)
        
        letters = ['A', 'B', 'C', '','',
                   'D', 'E', 'F', '','',
                   'G', 'H', 'I', 'J', 'K',
                   'L', 'M', 'N', 'O', 'P']
        for i in range(4):
            for j in range(5):
                if not (i<2 and j>=3):
                    axs[i,j].text(-0.15, 0.96, letters[5*i + j], transform=axs[i,j].transAxes, 
                            size=fontsize+4, weight='bold')
        
        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'individual_plant_for_paper.png')
        fig.savefig(save_name,dpi=400)
        plt.show()
        plt.close() 

        #%% Makes overview plot for a single plant analysis for the Supplemental information
        fig = plt.figure(figsize=(15,23))
        gs = gridspec.GridSpec(23,15)
        axs = np.empty((7,5), dtype=object)
        
        axs[0,0] = fig.add_subplot(gs[0:5,0:5])
        axs[0,1] = fig.add_subplot(gs[0:5,5:10])
        axs[0,2] = fig.add_subplot(gs[0:5,10:15])
        
        for i in range(5):
            axs[1,i] = fig.add_subplot(gs[5:8,3*i:3*i+3])
            axs[2,i] = fig.add_subplot(gs[8:11,3*i:3*i+3])
            axs[3,i] = fig.add_subplot(gs[11:14,3*i:3*i+3])
            axs[4,i] = fig.add_subplot(gs[14:17,3*i:3*i+3])
        
        axs[5,0] = fig.add_subplot(gs[17:20,0:3])
        axs[6,0] = fig.add_subplot(gs[20:23,0:3])
        
        axs[5,1] = fig.add_subplot(gs[17:23,3:9])
        axs[5,2] = fig.add_subplot(gs[17:23,9:15])
        fontsize = 14
        
        axs[0,0].fill_between(Time_model, A_low, A_high, color='orange', alpha=0.2, label='95% model CI')
        axs[0,0].plot(Time_model, A_median, color='orange', label="Modelled")
        A_fig, = axs[0,0].plot(Time, Assimilation*1e6, ls='', marker='o', color='orange', label='Measured')
        axs[0,0].set_ylabel(r'Net $CO_2$ assimilation $[\mu mol\ m^{-2}\ s^{-1}]$', fontsize=fontsize)
        axs[0,0].set_ylim(Assimilation.max()*1e6 - 1,Assimilation.max()*1e6 + 0.15)
        
        line = Line2D([], [], color="orange")
        band = Patch(facecolor="orange", alpha=0.2)
        axs[0,0].legend(
            [A_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower right', fontsize=fontsize  )
                    

        axs[0,1].plot(Time_model, qT_med , label='Modelled', color='blue')
        axs[0,1].fill_between(Time_model, qT_low, qT_high, color='blue', alpha=0.2, label='95% model CI')
        qT_fig, = axs[0,1].plot(Time_qT, 
                              qT_observed_all ,
                              c='blue', label='Measured', marker='o', ls="", markersize=3)
        axs[0,1].set_ylabel('qT (%)', fontsize=fontsize)
        axs[0,1].set_ylim(-1, qT_observed_all.max() +2)
        line = Line2D([], [], color="blue")
        band = Patch(facecolor="blue", alpha=0.2)
        axs[0,1].legend(
            [qT_fig, (line, band)],
            ['Measured', "Modelled (95% CI)"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0.01, 0.94),
            fontsize=fontsize         )
        

        QY = D_P700_median
        QY_low = D_P700_low
        QY_high = D_P700_high
        axs[0,2].plot(Time_model,1- QY ,c='black', ls='--', label='P700 donor limitation')
        axs[0,2].fill_between(Time_model, 1-QY_low, 1-QY_high, color='black', alpha=0.2, label='95% credibility interval')
        axs[0,2].plot(Time_model, QY ,c='orange',ls='--', label='P700 photochemistry')
        axs[0,2].fill_between(Time_model, QY_low, QY_high, color='orange', alpha=0.2, label='95% credibility interval')

        
        donor_side_limitation_uncorrected = (plant_test.dual_pam_data.Ps-plant_test.dual_pam_data.P0)/plant_test.Pm
        QY_uncorrected = (plant_test.dual_pam_data.Phigh-plant_test.dual_pam_data.Ps)/plant_test.Pm
        donor_side_limitation = donor_side_limitation_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        QY = QY_uncorrected / (donor_side_limitation_uncorrected + QY_uncorrected)
        
        
        donor_fig, = axs[0,2].plot(plant_test.dual_pam_data.minutes, donor_side_limitation,
                    c='black', label='Donor side limitation', ls='', marker='o')

        QY_fig, = axs[0,2].plot(plant_test.dual_pam_data.minutes, QY,
                    c='orange', label='Photochemistry', ls='', marker='o')

        axs[0,2].set_ylabel(r'$\phi/(\phi_I + \phi_{ND})$', fontsize=fontsize)
        axs[0,2].set_ylim(-0.1,1.5)
        axs[0,2].legend(loc='upper left', fontsize=fontsize)
        

        modelled_line = Line2D([],[], c='black', ls='--' )
        modelled_band = Patch(facecolor='black', alpha=0.2)
        QY_patch = Patch(facecolor='orange', alpha=1)
        donor_patch = Patch(facecolor='black', alpha=1)


        axs[0,2].legend(
            [QY_fig, modelled_line, modelled_band, QY_patch, donor_patch],
            [r'Measured', r"Modelled", '95% CI',
             'Photochemistry', "Donor limitation"],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='upper left', bbox_to_anchor=(0, 0.96),
            ncols=2, fontsize=fontsize-1
        )

        
        for i in range(3):
            axs[0,i].set_xlabel('Time (min)', fontsize=fontsize)
            axs[0,i].set_xlim(30,max_t)
            axs[0,i].tick_params(axis='both', which='major', labelsize=15)
            for j in range(1,4):
                
                banner = mpatches.Rectangle(((light_changes[j]-light_changes[1])/(max_t-light_changes[1]), 0.94), (light_changes[j+1]-light_changes[j])/(max_t-light_changes[1]), 0.06,
                                            transform=axs[0,i].transAxes,  # axes coords
                                            facecolor=colors[j],
                                            edgecolor=colors[j], linewidth=0.8)
                axs[0,i].add_patch(banner)
                axs[0,i].text( ((light_changes[j+1]+light_changes[j])-2*light_changes[1])/(2*(max_t-light_changes[1])), 0.975, texts[j], transform=axs[0,i].transAxes,
                        va="center_baseline", ha="center", fontsize=15, color="black")
                axs[0,i].vlines(light_changes[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
        #_____________________________________________________________________________
        def exp_transform_ST_rate(x):
            return np.exp(x)*60
        
        def exp_transform_milion(x):
            return np.exp(x)*1e6
        
        def exp_transform_thousand(x):
            return np.exp(x)*1e3
        
        def find_bayesian_stats_string(param):
            bulk_str =   str(int(summary_sel.at[param, 'ess_bulk'])) + '\n' #r'$ess_{bulk}$ = ' +
            tail_str =  str(int(summary_sel.at[param, 'ess_tail'])) + '\n' #r'$ess_{tail}$ = ' + 
            r_hat_str =  str(summary_sel.at[param, 'r_hat']) # r'$\hat{R}$ = ' + 
            string = bulk_str + tail_str + r_hat_str
            return string
        
        def find_first_bayesian_stats_string(param):
            bulk_str =  r'$ess_{bulk}$ = ' + str(int(summary_sel.at[param, 'ess_bulk'])) + '\n'
            tail_str = r'$ess_{tail}$ = ' +  str(int(summary_sel.at[param, 'ess_tail'])) + '\n' 
            r_hat_str = r'$\hat{R}$ = ' +  str(summary_sel.at[param, 'r_hat']) 
            string = bulk_str + tail_str + r_hat_str
            return string
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        param_list = ['Rd', 'PSI_PSII_ratio', 'CB6F_recovery_rate', 'forward_ST_rate', 
                      'reverse_ST_rate', 'Chl_per_area', 'PS_per_chl', 'chl_ab',
                      'K_N2', 'g_meso',
                      'eta_LET', 'eta_CET', 'gamma_star', 'K_P2', 'K_P1',
                      'Vcmax', 'PQ_per_chl', 'MM_constant_oxy', 'MM_constant_carboxy', 'K_D2',
                      'leaf_abs_change', 'sigma']
        factor_list = [1e6,1,1e6,1,1,1e3, 1e3,1,1,1,1,1,1e6,1,1,1e6,1e3,1e3,1e6,1,1,1]
        xlabels = [r'$R_{day}$ $[\mu mol\ m^{-2}\ s^{-1}]$', 'PSI/PSII ratio [-]',
                   'maximal CB6F recovery rate $[\mu mol\ m^{-2}\ s^{-1}]$',
                   r'$k_{1\rightarrow2}$ $[min^{-1}]$',r'$k_{2\rightarrow1}$ $[min^{-1}]$',
                   r'Chl content $[mmol\ m^{-2}]$', r'PS content $[mmol\ mol^{-1}\ chl]$',
                   'chl a/b ratio [-]', r'$K_{N,II}$ $[ns^{-1}]$', r'$g_m$ [$mol\ m^{-2}\ s^{-1}\ bar^{-1}]$',
                   r'$n_{LET}$ [mol ATP $mol^{-1}\ e^-]$', r'$n_{CET}$ [mol ATP $mol^{-1}\ e^-]$',
                   r'$\Gamma^*$ $[\mu bar]$', r'$K_{P,II}$ $[ns^{-1}]$',
                   r'$K_{P,I}$ $[ns^{-1}]$', r'$V_{cmax}$ $[\mu mol\ m^{-2}\ s^{-1}]$',
                   r'PQ content $[mmol\ mol^{-1} \ chl]$', '$K_{mO}$ [mbar]',
                   r'$K_{mC}$ $[\mu bar]$', r'$K_{D,II}$ $[ns^{-1}]$',
                   'leaf absorptance shift [-]', r'remaining error $[\mu mol\ m^{-2}\ s^{-1}]$']
        prior_list = ['R_d', 'PSI_PSII_ratio', 'CB6F_recovery_rate', 'forward_ST_rate',
                      'reverse_ST_rate', 'Chl_per_area', 'PS_per_chl', 'Chl_a_b',
                      'K_N2', 'g_m',
                      'eta_LET', 'eta_CET', 'gamma_star', 'K_p2', 'K_p1',
                      'Vcmax', 'PQ_per_chl', 'MM_constant_oxy', 'MM_constant_carboxy','K_D2',
                      'alpha leaf', None
                      ]

        
        for i in [1,2,3,4]:
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
                    
                
                elif factor_list[var_num]==1:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0], 
                                                             priors.at[prior_list[var_num], 'prior'][1])
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x, pdf, label='prior', c='black', ls='--')
                    
                elif factor_list[var_num]==1e3:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_thousand, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')
                    if param_list[var_num]=='Chl_per_area':
                        params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0]*1e-6, 
                                                                 priors.at[prior_list[var_num], 'prior'][1]*1e-6)
                    elif param_list[var_num]=='MM_constant_oxy':
                        params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0]*1e-3, 
                                                                 priors.at[prior_list[var_num], 'prior'][1]*1e-3)
                    
                    else:
                        params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0], 
                                                                 priors.at[prior_list[var_num], 'prior'][1])
                    x = np.linspace(
                        lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
                        lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
                        500)
                    pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
                    axs[i,j].plot(x*1e3, pdf*1e-3, label='prior', c='black', ls='--')
                else:
                    az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                                          transform=exp_transform_milion, label='posterior',
                                          point_estimate=None, round_to=2, hdi_prob=0.95,
                                          c='black')

                    params = solve_for_log_normal_parameters(priors.at[prior_list[var_num], 'prior'][0]*1e-6, 
                                                             priors.at[prior_list[var_num], 'prior'][1]*1e-6)
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

                axs[i,j].set_xlabel(xlabels[var_num], fontsize=fontsize)
                if var_num != 0:
                    axs[i,j].text(1, 0.98, find_bayesian_stats_string(param_list[var_num]), va='top',
                                 ha='right', fontsize=fontsize-1, 
                                 transform=axs[i,j].transAxes, bbox=props)
                else:
                    axs[i,j].text(1, 0.98, find_first_bayesian_stats_string(param_list[var_num]), va='top',
                                 ha='right', fontsize=fontsize-1, 
                                 transform=axs[i,j].transAxes, bbox=props)
        
        var_num, i, j =20, 5, 0
        az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j], label='posterior',
                              point_estimate=None, round_to=2, hdi_prob=0.95,
                              c='black')

        x = np.linspace(
            norm.ppf(0.001, loc=priors.at[prior_list[var_num], 'prior'][0], scale=priors.at[prior_list[var_num], 'prior'][1]),
            norm.ppf(0.999, loc=priors.at[prior_list[var_num], 'prior'][0], scale=priors.at[prior_list[var_num], 'prior'][1]),
            500)
        pdf = norm.pdf(x, loc=priors.at[prior_list[var_num], 'prior'][0], scale=priors.at[prior_list[var_num], 'prior'][1])
        axs[i,j].plot(x, pdf, label='prior', c='black', ls='--')
        axs[i,j].set_xlabel(xlabels[var_num], fontsize=fontsize)
        axs[i,j].set_xlim(-0.1,0.1)
        axs[i,j].text(0.98, 0.98, find_bayesian_stats_string(param_list[var_num]), va='top',
                     ha='right', fontsize=fontsize-1, 
                     transform=axs[i,j].transAxes, bbox=props)
        
        var_num, i, j =21, 6, 0
        az.plot_posterior(idata_sel,var_names=param_list[var_num], ax = axs[i,j],
                              transform=exp_transform_milion, label='posterior',
                              point_estimate=None, round_to=2, hdi_prob=0.95,
                              c='black')

        params = solve_for_log_normal_parameters(0.1e-6, 0.1e-6)
        x = np.linspace(
            lognorm.ppf(0.001, s=params[1], scale=np.exp(params[0])),
            lognorm.ppf(0.999, s=params[1], scale=np.exp(params[0])),
            500)
        pdf = lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
        axs[i,j].plot(x*1e6, pdf*1e-6, label='prior', c='black', ls='--')
        axs[i,j].set_xlabel(xlabels[var_num], fontsize=fontsize)
        axs[i,j].text(0.98, 0.98, find_bayesian_stats_string(param_list[var_num]), va='top',
                     ha='right', fontsize=fontsize-1, 
                     transform=axs[i,j].transAxes, bbox=props)
        
        for i in [5,6]:
            axs[i,0].set_title('')
            axs[i,0].legend([])
            for txt in list(axs[i,0].texts):
                if "HDI" in txt.get_text():
                    txt.remove()
    

        idata_sel.posterior["LHCII_per_PSI"].quantile(0.5, dim=["draw", "chain"]).values
        
        
        
        axs[1,1].legend(fontsize=fontsize-1, loc='upper left')
        
        plot_predictives = True
        if plot_predictives:
            prior_pred_low = idata_prior_pred.prior_predictive['Y_obs'].quantile(0.025, dim=["draw", "chain"]).values*1e6
            prior_pred_med = idata_prior_pred.prior_predictive['Y_obs'].quantile(0.5, dim=["draw", "chain"]).values*1e6
            prior_pred_high = idata_prior_pred.prior_predictive['Y_obs'].quantile(0.975, dim=["draw", "chain"]).values*1e6
            
            axs[5,1].plot(Time_obs, prior_pred_med, c='k', label='median prior predictive')
            axs[5,1].fill_between(Time_obs, prior_pred_low, prior_pred_high, alpha=0.2, color='black',
                                  label='95% prior predictive')
            axs[5,1].scatter(Time_obs, idata_prior_pred.observed_data['Y_obs']*1e6, color='orange',
                             label='observed data')
            axs[5,1].set_ylabel(r'Prior predictive assimilation $[\mu mol\ m^{-2}\ s^{-1}]$',
                                fontsize=fontsize)
            axs[5,1].set_ylim(prior_pred_low.min(),prior_pred_high.max()+0.2)
            
            
            post_pred_low = idata_posterior_pred['Y_obs'].quantile(0.025, dim=["draw", "chain"]).values*1e6
            post_pred_med = idata_posterior_pred['Y_obs'].quantile(0.5, dim=["draw", "chain"]).values*1e6
            post_pred_high = idata_posterior_pred['Y_obs'].quantile(0.975, dim=["draw", "chain"]).values*1e6

            axs[5,2].plot(Time_obs, post_pred_med, c='k', label='median posterior predictive')
            axs[5,2].fill_between(Time_obs, post_pred_low, post_pred_high, alpha=0.2, color='black',
                                  label='95% CI posterior predictive')
            axs[5,2].scatter(Time_obs, idata_prior_pred.observed_data['Y_obs']*1e6, color='orange',
                             label='observed data')
            axs[5,2].set_ylabel(r'Posterior predictive assimilation $[\mu mol\ m^{-2}\ s^{-1}]$',
                                fontsize=fontsize)
            axs[5,2].set_ylim(post_pred_low.min(),post_pred_high.max()+0.2)
                    
            for i in range(2):
                axs[5,i+1].set_xlabel('Time (min)', fontsize=fontsize)
                axs[5,i+1].set_xlim(46,78)
                axs[5,i+1].tick_params(axis='both', which='major', labelsize=fontsize-2)
                axs[5,i+1].legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(0.99,0.94))
                l = [46,62,78]

                for j in range(2):
                    banner = mpatches.Rectangle(((l[j]-l[0])/(l[2]-l[0]), 0.94), (l[j+1]-l[j])/(l[2]-l[0]), 0.06,
                                                transform=axs[5,i+1].transAxes,  # axes coords
                                                facecolor=colors[j],
                                                edgecolor=colors[j], linewidth=0.8)
                    axs[5,i+1].add_patch(banner)
                    axs[5,i+1].text( ((l[j+1]+l[j])-2*l[0])/(2*(l[2]-l[0])), 0.975, texts[j], transform=axs[5,i+1].transAxes,
                            va="center_baseline", ha="center", fontsize=fontsize, color="black")
                    axs[5,i+1].vlines(l[j],-10,15, ls='--', color='lightgrey', zorder=0)
        else:
            axs[5,1].text(0.5,0.5,'Placeholder for prior predictive', 
                          va='center', ha='center')
            axs[5,2].text(0.5,0.5,'Placeholder for poseterior predictive', 
                          va='center', ha='center')

        
        fig.tight_layout()
        save_name = os.path.join(output_folder,'individual_plant_for_SI.png')
        fig.savefig(save_name,dpi=400)
        plt.show()
        plt.close() 
        
        
        
    # #%% Extra scenario figure for quantifying effect of ST
            
    #     def take_time_step_ST_effect (Ci, light_spect,  A_prev, chloroplastic_carbon, LHCII_per_PSII, 
    #                         LHCII_per_PSI, D_P680, D_P700, PQH2, Fm, Fm_2, 
    #                         P680_excitation, P700_excitation,
    #                         J_P680_forward, J_P700_forward,
    #                         dephosphorylation_rate_constant,
    #                         phosphorylation_rate_constant, Vcmax, Rd, eta_LET,
    #                         eta_CET, gamma_star, MM_constant_oxy, MM_constant_carboxy,
    #                         CB6F_recovery_rate, g_meso, K_P2, K_P1, 
    #                         D_P680_tot, D_P700_tot, PQ_pool,
    #                         rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
    #                         open_sharing_rate, closed_sharing_rate, leaf_absorbance,
    #                         timestep):
        
    #         D_P680_abs = D_P680 * D_P680_tot
    #         D_P700_abs = D_P700 * D_P700_tot
    #         PQH2_abs = PQH2 * PQ_pool
            
    #         #calculate eta
    #         a = 3 + 7 * (gamma_star/chloroplastic_carbon)
    #         b = 4 + 8 * (gamma_star/chloroplastic_carbon) * eta_CET
    #         eta = 1 - (eta_LET / eta_CET) + (a / b)
            
    #         # update_spectra()
    #         PSI_LHCII_spectrum_model = D_P700_tot * LHCII_per_PSI * spectrum_LHCII
    #         PSI_LHCI_absorbance_spectrum = D_P700_tot * spectrum_PSI_LHCI
    #         absorptance_PSI = PSI_LHCII_spectrum_model + PSI_LHCI_absorbance_spectrum     
            
    #         PSII_core_spectrum_model = D_P680_tot * spectrum_CSM
    #         PSII_LHCII_spectrum_model = D_P680_tot * (LHCII_per_PSII-2) * spectrum_LHCII
    #         absorptance_PSII = PSII_core_spectrum_model + PSII_LHCII_spectrum_model 
            
    #         # calc_P680_excitation()
    #         sigma_II =  absorptance_PSII/ (absorptance_PSI + absorptance_PSII)
    #         P680_excitation = (light_spect * leaf_absorbance * sigma_II).sum(axis=0)
        
    #         # calc_P700_excitation()
    #         sigma_I = absorptance_PSI/ (absorptance_PSI +  absorptance_PSII)
    #         P700_excitation = (light_spect * leaf_absorbance * sigma_I).sum(axis=0)
    #         for i in range(3):
    #             # phi_U2 calculation
    #             phi_U2 = (D_P680_abs / D_P680_tot) * open_sharing_rate + (1 - (D_P680_abs / D_P680_tot)) * closed_sharing_rate
    #             # PQH2 calculation
    #             Phi_P680_photo = K_P2 / rate_tot_P680_open
    #             a_val = PQ_pool * Phi_P680_photo * P680_excitation * eta  
    #             b_val = (CB6F_recovery_rate * (1 - phi_U2)) + (Phi_P680_photo * P680_excitation * eta)
    #             PQH2_abs = a_val / b_val
            
    #             # D_P700_open calculation
    #             Phi_P700_photo = K_P1 / rate_tot_P700_open
    #             D_P700_abs = (PQH2_abs * CB6F_recovery_rate * D_P700_tot) / (PQ_pool * Phi_P700_photo * P700_excitation)
            
    #             # D_P680_open calculation
    #             D_P680_abs = (1 - (PQH2_abs / PQ_pool)) * D_P680_tot
             
    #             D_P680_abs = np.where(D_P700_abs > D_P700_tot,
    #                                ((Phi_P700_photo * P700_excitation) / eta) * ((1 - phi_U2) / (Phi_P680_photo * P680_excitation)) * D_P680_tot,
    #                                D_P680_abs)
            
    #             PQH2_abs = np.where(D_P700_abs > D_P700_tot, (1 - (D_P680_abs / D_P680_tot)) * PQ_pool, PQH2_abs)
    #             D_P700_abs = np.min(np.array([D_P700_tot, D_P700_abs]))
        
    #         #calc_J_P680_forward()
    #         Phi_P680_photo = K_P2 / rate_tot_P680_open
    #         J_P680_forward = D_P680_abs * Phi_P680_photo * P680_excitation / (D_P680_tot * (1 - phi_U2))
        
    #         #calc_J_P700_forward()
    #         Phi_P700_photo = K_P1 / rate_tot_P700_open
    #         J_P700_forward = D_P700_abs * Phi_P700_photo * P700_excitation / D_P700_tot 
        
    #         for i in range(3):
    #             W_J = (chloroplastic_carbon * J_P680_forward) / (4 * chloroplastic_carbon + 8 * gamma_star)
    #             W_c = (chloroplastic_carbon * Vcmax) / (chloroplastic_carbon + MM_constant_carboxy * (1 + (chloroplastic_oxygen / MM_constant_oxy)))
    #             A = (1 - (gamma_star / chloroplastic_carbon)) * np.min(np.array([W_c, W_J])) - Rd
    #             chloroplastic_carbon = Ci - (A / g_meso)
            
            
    #         #updating LHCII distribution
    #         LHCII_dephosphorylated = LHCII_per_PSII * D_P680_tot 
    #         LHCII_phosphorylated = LHCII_per_PSI * D_P700_tot 
    #         dephosphorylation_rate = LHCII_phosphorylated * dephosphorylation_rate_constant
    #         phosphorylation_rate = LHCII_dephosphorylated * phosphorylation_rate_constant * (PQH2_abs / PQ_pool)
            
    #         change_rate = (phosphorylation_rate - dephosphorylation_rate) * timestep
    #         LHCII_phosphorylated += change_rate 
    #         LHCII_dephosphorylated -= change_rate 
        
                
    #         LHCII_per_PSII_new = LHCII_dephosphorylated / D_P680_tot
    #         LHCII_per_PSI_new = LHCII_phosphorylated / D_P700_tot
            
            
    #         #calculations for qT
    #         excitation_PSII_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_II).sum(axis=0)
    #         excitation_PSI_PAM = (PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_I).sum(axis=0)
            
    #         a = excitation_PSII_PAM * epsilon_F2 * (K_F2 / (rate_tot_P680_closed-K_U2))
    #         b = excitation_PSI_PAM * epsilon_F1 * (K_F1 / rate_tot_P700_closed)
    #         Fm = a + b
    #         Fm_2 = a
            
    #         D_P680 = D_P680_abs / D_P680_tot
    #         D_P700 = D_P700_abs / D_P700_tot
    #         PQH2 = PQH2_abs / PQ_pool
            
    #         new_state = (A, chloroplastic_carbon, LHCII_per_PSII_new, LHCII_per_PSI_new, D_P680, D_P700, PQH2, Fm, Fm_2, P680_excitation, P700_excitation, J_P680_forward, J_P700_forward)
        
    #         return new_state
    
    #     run_extra_quantification=True
    #     if run_extra_quantification:
                    
    #         output_folder = 'theoretical figures/'
    #         #define parameters
        
    #         gamma_star = float(np.exp(idata_sel.posterior['gamma_star'].mean())) #
    #         MM_constant_oxy = float(np.exp(idata_sel.posterior['MM_constant_oxy'].mean())) # 
    #         MM_constant_carboxy =float(np.exp(idata_sel.posterior['MM_constant_carboxy'].mean()))#
    #         chloroplastic_oxygen = 212.7*1e-3 # 
    #         g_meso = float(np.exp(idata_sel.posterior['g_meso'].mean())) #mol m-2 s-1 bar-1 
    #         Vcmax = float(np.exp(idata_sel.posterior['Vcmax'].mean()))# random find source
    #         Rd = float(np.exp(idata_sel.posterior['Rd'].mean())) #random find source
        
    #         Ci = 350*1e-6
            
    #         #set rate parameters in PSI and PSII
          
    #         K_P2 = float(np.exp(idata_sel.posterior['K_P2'].mean())) # ns-1, rate of photochemistry
    #         K_N2 =  float(np.exp(idata_sel.posterior['K_N2'].mean())) # ns-1, assumed at 0 under light limiting conditions
    #         K_D2 = float(np.exp(idata_sel.posterior['K_D2'].mean())) # ns-1, rate of constitutive heat loss
    #         K_F2 = 0.05 # ns-1, rate of fluorescence
    #         K_U2 = 2.1 # ns-1, to do
            
    #         K_P1 =  float(np.exp(idata_sel.posterior['K_P1'].mean())) # ns-1, rate of photochemistry
    #         K_D1 = 0.55 # ns-1, rate of constitutive heat loss
    #         K_F1 = 0.05 # ns-1, rate of fluorescence
    #         K_X1 = K_P1 # ns-1, rate of dissipation in closed P700, set equal to K_P1
    #         rate_tot_P700_open = K_P1 + K_D1 + K_F1
    #         rate_tot_P700_closed = K_X1 + K_D1 + K_F1
        
    #         epsilon_F1 = 2
    #         epsilon_F2 = 1
            
    #         # Set initial number of thylakoid membrane components.
        
    #         Chl_per_area =  float(np.exp(idata_sel.posterior['Chl_per_area'].mean())) # mol m-2 
    #         PS_per_chl =  float(np.exp(idata_sel.posterior['PS_per_chl'].mean())) # mol/mol chl, Ibrahim 2022,
    #         PSI_PSII_ratio =  float(np.exp(idata_sel.posterior['PSI_PSII_ratio'].mean())) #PSI/PSII, McKenzie 2020
        

    #         CB6F_recovery_rate=  float(np.exp(idata_sel.posterior['CB6F_recovery_rate'].mean())) # mol/mol chl, kruk 2006

    #         PQ_per_chl =  float(np.exp(idata_sel.posterior['PQ_per_chl'].mean())) # mol/mol chl, kruk 2006

    #         chl_ab =  float(np.exp(idata_sel.posterior['chl_ab'].mean())) #random value needs source
            
    #         # State number of chlorophylls per protein complex.
    #         PS_table = pd.DataFrame({'chl_a': [148, 108, 24],
    #                                  'chl_b': [12, 49, 18],
    #                                  'chl_tot': [160, 157, 42]}, index=['PSI-LHCI', 'CSM', 'LHCII'])
    #         #PSI-LHCI from Pan 2018; CSM from Su 2014; LHCII from Laisk 2014
            
    #         k_CB6F_star = 300 # mol PQH2 mol-1 sites s-1, catalytic constant for Cyt B6F assuming relaxed photosynthetic control
            
    #         eta_LET =  float(np.exp(idata_sel.posterior['eta_LET'].mean())) # mol ATP mol-1 e-1
    #         eta_CET =  float(np.exp(idata_sel.posterior['eta_CET'].mean())) # mol ATP mol-1 e-1
            
                
    #         #import leaf absorbance and absorbance by photosynthetic pigments. 
    #         parameters = pd.read_excel('model parameters fvCBwd model/parameters.xlsx', header=2)
    #         # Inter- and extrapolates the missing values in the absorption by non-photosynthetic pigmens.  
    #         known_abs_pig = np.transpose(np.array(parameters[parameters['alpha_pigments'].notna()][['Wavelength(nm)', 'alpha_pigments']]))
    #         x_new = np.array(parameters['Wavelength(nm)'])
    #         parameters['alpha_pigments'] = np.interp(x_new, known_abs_pig[0], known_abs_pig[1])
    #         absorbance_photopigments = np.array(parameters.alpha_pigments)
    #         leaf_absorbance_lit = np.array(parameters.leaf_absorbance)
                
    #         test_plant = 'NT9'
    #         with open(f'pickles_with_priors/{test_plant}.pkl', 'rb') as file:
    #             plant_test = pkl.load(file)
    #             leaf_absorbance_NT = plant_test.leaf_absorbance

            
    #         #import absorbance spectra of protein components.
    #         file_names = ['Hogewoning_PSI', 'Hogewoning_PSII_core', 'Hogewoning_LHCII']
    #         spectra = np.zeros((3,371))
    #         for i in range(len(file_names)):
    #             file_name = 'model parameters FvCBwd model/' + file_names[i] + '.csv'
    #             file = pd.read_csv(file_name, sep=';', decimal=',', names=['wavelength', 'intensity'])
    #             spectrum = np.array(file.intensity)
    #             spectra[i] = spectrum
                
    #         PSI_LHCI_chl = PS_table.at['PSI-LHCI', 'chl_a'] + 0.7 * PS_table.at['PSI-LHCI', 'chl_b']
    #         spectrum_PSI_LHCI = (PSI_LHCI_chl / spectra[0][250:].sum(axis=0)) * (spectra[0])
            
    #         CSM_chl = PS_table.at['CSM', 'chl_a'] + 0.7 * PS_table.at['CSM', 'chl_b']
    #         spectrum_CSM = (CSM_chl / spectra[1][250:].sum(axis=0)) * (spectra[1])
            
    #         LHCII_chl = PS_table.at['LHCII', 'chl_a'] + 0.7 * PS_table.at['LHCII', 'chl_b']
    #         spectrum_LHCII = (LHCII_chl / spectra[2][250:].sum(axis=0)) * (spectra[2])
             
        

    #         #initialise light and simulation time
    #         section_minutes = 15
    #         section_time = section_minutes * 60
    #         time_steps_model = 5
    #         model_length =int( 3 * section_time / time_steps_model)
    #         Ci_model = np.repeat([Ci], model_length)
    #         Time = np.arange(-1 * section_time/60,2 * section_time/60, time_steps_model/60)
        
    #         t_plot = (0, 2*section_time/60)
            
            
            
            
    #         light_list = np.zeros((model_length,371))
    #         for i in range(len(light_list)):
    #             if (i>=len(light_list)/3) and (i<len(light_list)*2/3):
    #                 light_list[i] = WL_spect*1e-6
    #             else:
    #                 light_list[i] = WL_FR_spect*1e-6
            
            
            
            
    #         LHCII_per_PSII_median = idata_sel.posterior["LHCII_per_PSII"].quantile(0.5, dim=["draw", "chain"]).values
    #         #Three LHCII distributions are chosen. WL equilibrium, WL+FR equilibrium and average
            
    #         LHCII_per_PSII_init_list = [3.92,
    #                                     idata_sel.posterior["LHCII_per_PSII"].quantile(0.5, dim=["draw", "chain"]).values[-1],
    #                                     4.389]
    #         LHCII_per_PSI_init_list = [1.10,
    #                                    idata_sel.posterior["LHCII_per_PSI"].quantile(0.5, dim=["draw", "chain"]).values[-1],
    #                                    0.63]
    #         forward_rate_list = [0,
    #                              float(np.exp(idata_sel.posterior['forward_ST_rate'].mean())),
    #                              0]
    #         reverse_rate_list = [0,
    #                              float(np.exp(idata_sel.posterior['reverse_ST_rate'].mean())),
    #                              0]
        
            
    #     def run_model(dephosphorylation_rate_constant,
    #                   phosphorylation_rate_constant,
    #                   LHCII_per_PSII_init,
    #                   LHCII_per_PSI_init,
    #                   cutoff = 3,
    #                   leaf_absorbance = leaf_absorbance_lit):
    #         #initialise thylakoid membrane
    #         concentration_PSII = PS_per_chl * (1 - (PSI_PSII_ratio / (1 + PSI_PSII_ratio))) # mol/mol Chl
    #         concentration_PSI = PS_per_chl - concentration_PSII # mol/mol Chl
    #         D_P680_tot = concentration_PSII * Chl_per_area # mol m-2
    #         D_P700_tot = concentration_PSI * Chl_per_area # mol m-2
            
    #         PQ_pool = PQ_per_chl * Chl_per_area # mol m-2
            
       
            
           
    #         #defining ST model parameters
    #         rate_tot_P680_open = K_P2 + K_N2 + K_D2 + K_F2 + K_U2
    #         rate_tot_P680_closed = K_N2 + K_D2 + K_F2 + K_U2
    #         open_sharing_rate = K_U2 / rate_tot_P680_open
    #         closed_sharing_rate = K_U2 / rate_tot_P680_closed
        
            
    #         #initalise initial state
    #         A = 0
    #         chloroplastic_carbon_init = Ci

    #         D_P680_init = 1 #D_P680 is initiated fully open
    #         D_P700_init = 1 #D_P680 is initiated fully open
    #         PQH2_init = 0
    #         Fm_init = 1 # placeholder for Fm, value is not used
    #         Fm_2_init = 1 # placeholder for Fm, value is not used
    #         P680_excitation_init = 40*1e-6 # placeholder for excitation, value is not used
    #         P700_excitation_init = 40*1e-6 # placeholder for excitation, value is not used
    #         J_P680_forward_init = 40*1e-6 # placeholder for flow rate, value is not used
    #         J_P700_forward_init = 40*1e-6 # placeholder for flow rate, value is not used
            
            
            
    #         results = np.zeros((model_length+1,13))
    #         results[0] = [A, chloroplastic_carbon_init, LHCII_per_PSII_init, LHCII_per_PSI_init,
    #                       D_P680_init, D_P700_init, PQH2_init, Fm_init, Fm_2_init, P680_excitation_init,
    #                       P700_excitation_init, J_P680_forward_init, J_P700_forward_init]


    #         for i in range(model_length):

    #             results[i+1,:] = take_time_step_ST_effect(Ci, light_list[i], results[i,0], results[i,1], results[i,2],
    #                                           results[i,3], results[i,4], results[i,5], results[i,6], results[i,7], results[i,8],
    #                                           results[i,9], results[i,10],
    #                                           results[i,11], results[i,12],
    #                                           dephosphorylation_rate_constant,
    #                                           phosphorylation_rate_constant, Vcmax, Rd, eta_LET,
    #                                           eta_CET, gamma_star, MM_constant_oxy, MM_constant_carboxy,
    #                                           CB6F_recovery_rate, g_meso, K_P2, K_P1, 
    #                                           D_P680_tot, D_P700_tot, PQ_pool,
    #                                           rate_tot_P680_open, rate_tot_P680_closed, rate_tot_P700_open, rate_tot_P700_closed,
    #                                           open_sharing_rate, closed_sharing_rate, leaf_absorbance,
    #                                           time_steps_model)
                
                
    #         #Convert to useful units
    #         results[:,0] = results[:,0]*1e6 #A
    #         results[:,1] = results[:,1]*1e6 #Cc
    #         results[:,9] = results[:,9]*1e6 #P580 excitation
    #         results[:,10] = results[:,10]*1e6 #P700 excitation
    #         results[:,11] = results[:,11]*1e6 #J_P680 forward
    #         results[:,12] = results[:,12]*1e6 #J_P700 forward
        
            
    #         index_Fm = int(model_length*1/3)+1
    #         qT = 100*((results[:,7][index_Fm] - results[:,7]) / results[:,7][index_Fm])
            
    #         results_frame = pd.DataFrame(results, 
    #                                      columns=['A', 'Cc', 'LHCII_per_PSII',
    #                                               'LHCII_per_PSI', 'D_P680', 'D_P700',
    #                                               'PQH2', 'Fm', 'Fm2', 'P680_excitation',
    #                                               'P700_excitation', 'J_P680_forward', 'J_P700_forward'])
    #         results_frame['qT'] = qT
    #         results_frame['Time'] = np.concatenate((np.array([0]), Time), axis=None)
    #         results_frame.drop([0,], inplace=True)
        
    #         results_frame = results_frame.iloc[int(len(results_frame)/cutoff):]
    #         return results_frame
        
    #     results_frames = []
    #     for i in range(3):
    #         print(i)
    #         results_frames.append(run_model(reverse_rate_list[i],
    #                                         forward_rate_list[i],
    #                                         LHCII_per_PSII_init_list[i],
    #                                         LHCII_per_PSI_init_list[i]))
        
    #     #%%plot results
    #     fig, ax = plt.subplots(1,1,figsize=(6,4))
    #     labels = ['WL+FR LHCII distribution', 'state transitions active', 'WL LHCII distribution']
    #     ls_list = ['--', ':', '-.']
    #     for i in range(3):
    #         ax.plot(results_frames[i].Time, results_frames[i].A, label=labels[i],
    #                 c='black', ls=ls_list[i])
    #         print(results_frames[i].A.mean()/results_frames[1].A.mean())
            
    #     l=[0,15,30]
    #     for j in range(2):
    #         banner = mpatches.Rectangle(((l[j]-l[0])/(l[2]-l[0]), 0.94), (l[j+1]-l[j])/(l[2]-l[0]), 0.06,
    #                                     transform=ax.transAxes,  # axes coords
    #                                     facecolor=colors[j],
    #                                     edgecolor=colors[j], linewidth=0.8)
    #         ax.add_patch(banner)
    #         ax.text( ((l[j+1]+l[j])-2*l[0])/(2*(l[2]-l[0])), 0.975, texts[j], transform=ax.transAxes,
    #                 va="center_baseline", ha="center", fontsize=fontsize, color="black")
    #         ax.vlines(l[j],-10,15, ls='--', color='lightgrey', zorder=0)
        
    #     ax.set_xlim(0,30)
    #     ax.set_ylim(2.8,3.8)
    #     ax.set_xlabel('Time [min]')
    #     ax.set_ylabel(r'Net assimilation [$\mu mol\ m^{-2}\ s^{-1}$]')
    #     ax.legend()
        
    #     fig.tight_layout()
    #     save_name = os.path.join(output_folder,'ST_effect.png')
    #     fig.savefig(save_name,dpi=400)
    #     plt.show()
    #     plt.close() 
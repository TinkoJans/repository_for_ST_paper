# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:59:54 2025

@author: jans008
"""

#%% Aci fit
def analyse_Aci(Aci_data, Vcmax, Rd, plantID):
    gamma_star = 37.4 # ubar at 25C, standard value from Duursma 2015 and Bernacchi 2001  
    Kmc = 272.4 # ubar at 25C,  Duursma 2015 and Bernacchi 2001
    Oc = 212.7 # mbar, 21% of atmospheric pressure
    Kmo = 165.8 # mbar at 25C, Duursma 2015 and Bernacchi 2001
    alpha = 0.8 # Farquhar and Wong 1984
    f = 0.23 # Farquhar and Wong 1984
    theta = 0.7
    I_aci = 1000 #TODO check light intensity of the Aci
    Cc_aci = np.array(Aci_data.Cc)
    Assimilation_aci = np.array(Aci_data.A)
    
    def calc_photo(Cc, Vcmax, Jmax, Rd, PAR):
        J_light = 0.5 * (1-f) * alpha * PAR
        #J = (J_light*Jmax)/ (J_light+Jmax) 
        J = (J_light + Jmax - ((J_light+Jmax)**2 - 4*theta*J_light*Jmax) **0.5) / (2 * theta)
        W_J = (Cc*J) / (4*Cc+8*gamma_star)
        W_c = (Cc*Vcmax) / (Cc + Kmc * (1 + (Oc/Kmo)))
        photo = (1-(gamma_star/Cc)) * np.minimum(W_J, W_c) - Rd
        return photo
    
    
    aci_model = pm.Model()
    with aci_model:
        # Priors for unknown model parameters
        Vcmax=pm.TruncatedNormal('Vcmax', mu=Vcmax[0], sigma=Vcmax[1], lower=0.0)
        Jmax=pm.TruncatedNormal('Jmax', mu=250, sigma=150, lower=0.0)
        Rd=pm.TruncatedNormal('Rd', mu=Rd[0], sigma=Rd[1], lower=0.0)
        sigma = pm.HalfNormal("sigma", sigma=10)
    
        # Expected value of outcome
        J_light = 0.5 * (1-f) * alpha * I_aci
        #J = (J_light*Jmax)/ (J_light+Jmax) 
        J = (J_light + Jmax - ((J_light+Jmax)**2 - 4*theta*J_light*Jmax)**0.5 ) / (2 * theta)
        W_J = (Cc_aci*J) / (4*Cc_aci+8*gamma_star)
        W_c = (Cc_aci*Vcmax) / (Cc_aci + Kmc * (1 + (Oc/Kmo)))
        photo = (1-(gamma_star/Cc_aci)) * pm.math.minimum(W_J, W_c) - Rd
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=photo, sigma=sigma, observed=Assimilation_aci)
        
        # draw 1000 posterior samples
        idata_Aci = pm.sample(cores=1, chains=4)
        
        #plots all the posterior distributions with chains

        az.plot_trace(idata_Aci, combined=True)
        plt.savefig(f'Aci_curves/{plantID}_Aci_fit_chains', dpi=600)
        #prints a summary of the resulting posterior distributions
        az.summary(idata_Aci, round_to=2)
        
        
        posterior_Vcmax = idata_Aci.posterior["Vcmax"].values.flatten()
        posterior_Jmax = idata_Aci.posterior["Jmax"].values.flatten()
        posterior_Rd = idata_Aci.posterior["Rd"].values.flatten()
    
        Vcmax = (posterior_Vcmax.mean(), posterior_Vcmax.std(), 0)
        Rd = (posterior_Rd.mean(), posterior_Rd.std(), 0)
        Jmax = (posterior_Jmax.mean(), posterior_Jmax.std(), 0)
        
        #creates an overview plot of the fit with the data. 
        pm.sample_posterior_predictive(idata_Aci, extend_inferencedata=True, random_seed=rng)
        post = idata_Aci.posterior
        Cc_model = np.arange(Cc_aci.min(),Cc_aci.max(),1)
        phot_pred = calc_photo(xr.DataArray(Cc_model, dims=["obs_id"]), post['Vcmax'], post['Jmax'], post['Rd'], I_aci)
        fig, axs = plt.subplots(1,1, figsize=(6,5))
        axs.plot(Cc_model, phot_pred.mean(("chain","draw")),label='Modelled Aci-curve',  c='C1', alpha=0.6)
        az.plot_hdi(Cc_aci, idata_Aci.posterior_predictive['Y_obs'], ax=axs)
        axs.collections[-1].set_label("95% HDI")
        axs.scatter(Cc_aci, idata_Aci.observed_data['Y_obs'], label='observed')
        axs.set_xlabel(r'Cc $[\mu bar]$')
        axs.set_ylabel(r'Net assimilation $[\mu mol\ m^{-2}\ s^{-1}]$')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        summary_string = f'Vcmax:{Vcmax[0]:.0f}±{Vcmax[1]:.0f}\nRd:{Rd[0]:.2f}±{Rd[1]:.2f}\nJmax:{Jmax[0]:.2f}±{Jmax[1]:.2f}'
        axs.text(0.98,0.02, summary_string, transform=axs.transAxes, fontsize=11, 
                 bbox=props, ha='right',  va='bottom')
        axs.legend()
        plt.tight_layout()
        plt.savefig(f'Aci_curves/{plantID}_Aci_fit_bayes', dpi=600)
        

        
    return Vcmax, Rd



#%% Yin method
def analyse_R_day(Rd_data, Rd_prior, plantID):
    PAR = np.array(Rd_data.PAR)
    assimilation = np.array(Rd_data.assimilation)
    
    PAR_sel = PAR[2:6]    
    assimilation_sel = assimilation[2:6]
    
    def calc_photo(QY, PAR, Rd):
        photo = QY * PAR - Rd
        return photo
    
    def calc_error_propegation(mu1, sigma1, mu2, sigma2):
        mu_new = mu1 * mu2
        sigma_new = mu_new * math.sqrt((sigma1/mu1)**2+(sigma2/mu2)**2)
        return mu_new, sigma_new
    
    aci_model = pm.Model()
    with aci_model:
        # Priors for unknown model parameters
        QY=pm.TruncatedNormal('QY', mu=0.05, sigma=0.05, lower=0.0)
        Rd=pm.TruncatedNormal('Rd', mu=Rd_prior[0], sigma=Rd_prior[1], lower=0.0)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        
        

        
        # Expected value of outcome
        photo = QY * PAR_sel - Rd
    
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=photo, sigma=sigma, observed=assimilation_sel)
        
        # draw 1000 posterior samples
        idata = pm.sample(tune=2000, cores=1, chains=4, nuts={'target_accept':0.95})
        
        #plots all the posterior distributions with chains

        az.plot_trace(idata, combined=True)
        plt.savefig(f'Analysis_results/{plantID}/R_day_bayes_chains', dpi=600)
        #prints a summary of the resulting posterior distributions
        az.summary(idata, round_to=2)
        
        
        posterior_Rd = idata.posterior["Rd"].values.flatten()
        
        #factor 1.3 is added to increase for the underestimation of the Kok method. 
        #based on average and std of Yin 2011, zhen 2023 and sun 2023 measurements. 
        Rd_mean, Rd_std = calc_error_propegation(posterior_Rd.mean(), posterior_Rd.std(), 1.26, 0.27)
        Rd = (Rd_mean, Rd_std,0)
        
        posterior_QY = idata.posterior["QY"].values.flatten()
        QY = (posterior_QY.mean(), posterior_QY.std(),0)
        
        #creates an overview plot of the fit with the data. 
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        post = idata.posterior
        phot_pred = calc_photo(xr.DataArray(PAR, dims=["obs_id"]), post['QY'], post['Rd'])
        fig, axs = plt.subplots(1,1, figsize=(6,5))
        
        axs.plot(PAR, phot_pred.mean(("chain","draw")),label='Linear fit',  c='C1', alpha=0.6)
        az.plot_hdi(PAR_sel, idata.posterior_predictive['Y_obs'], ax=axs)
        axs.collections[-1].set_label("95% HDI")
        axs.plot(PAR, assimilation,c='red', marker='o', ls='', label='Excluded from fit')
        axs.plot(PAR_sel, assimilation_sel,c='green', marker='o', ls='', label='Included in fit')

        axs.set_xlabel(r'PAR intensity [$\mu$mol $m^{-2}$ $s^{-1}$]')
        axs.set_ylabel(r'Net $CO_2$ assimilation [$\mu$mol $m^{-2}$ $s^{-1}$]')
        axs.set_xlim(-5,160)
        axs.set_ylim(-2.2,7)
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        summary_string = f'QY:{QY[0]:.3f}±{QY[1]:.3f}\nRday:{posterior_Rd.mean():.2f}±{posterior_Rd.std():.2f}'
        axs.text(0.98,0.02, summary_string, transform=axs.transAxes, fontsize=11, bbox=props,
                 ha='right', va='bottom')
        axs.legend()
        plt.tight_layout()
        plt.savefig(f'analysis_results/{plantID}/R_day_bayes', dpi=600)
        plt.close()  

    return Rd


#%% tech reps of chl content
#Chlorophyll content calculations
def calc_stds_chlorophyll():
    
    file_folder = 'chlorophyll_content_measurements/tech reps'
    file_list = [filename for filename in os.listdir(file_folder)]
    global chl_tot_list
    global chl_ab_list
    chl_tot_list = []
    chl_ab_list = []
    for file_location in file_list:
        file_location = os.path.join(file_folder, file_location)    
        with open(file_location, 'r') as file:
            # Skip the first 14 rows (header) in the ASCII file
            lines = file.readlines()[14:]
            data = np.array([list(map(lambda x: float(x.replace(',', '.')), line.split())) for line in lines])
            
            chl_wavelength = data.T[0]
            chl_absorbance = data.T[1]
    
        chl_concentration_DMSO = calc_chl_concentration(chl_wavelength, chl_absorbance)
        leaf_area = calc_circle_area(10) * 2  # in m2
        
        chl_content_leaf = chl_concentration_DMSO * 5 * 2 / leaf_area

        chl_tot_list.append(chl_content_leaf[2])
        chl_ab_list.append(chl_content_leaf[0]/chl_content_leaf[1])
        
    chl_tot_std = np.array(chl_tot_list).std()
    chl_ab_std = np.array(chl_ab_list).std()
    return chl_tot_std, chl_ab_std


#%% k_CB6F analysis of uncertainty
def create_normal_distribution(mu, std, n):
    # takes mean (mu) and standard deviation (std) of the distribution
    # and the number of samples to draw (n)
    # returns a list of drawn samples from the distribution
    param_dist = np.random.normal(loc=mu, size=n, scale=std)
    return param_dist

def create_uniform_distribution(lb, ub, n):
    # takes mean (mu) and standard deviation (std) of the distribution
    # and the number of samples to draw (n)
    # returns a list of drawn samples from the distribution
    param_dist = np.random.uniform(low=lb, high=ub, size=n)
    return param_dist

def MM_rate_calc(PAR_int, max_rate, MM_constant):
    return max_rate * (PAR_int / (MM_constant + PAR_int))


def calc_CB6F_rate(CB6F_data, chl_prior, PS_ratio_prior, plantID):
    chl_content = create_normal_distribution(chl_prior[0], chl_prior[1], 10000) # umol m-2 s-1, measured on the tabaccos
    PS_per_chl_uncertainty = create_normal_distribution(0.0046, 0.00028, 10000) # mol/mol chl, Ibrahim 2022
    concentration_CB6F_uncertainty = create_normal_distribution(0.00076, 0.00011, 10000) # mol/mol Chl, ibrahim 2022
    PSI_PSII_ratio = create_normal_distribution(PS_ratio_prior[0], PS_ratio_prior[1], 10000) # PSI-core / PSII-core, will be measured, for now set to 1
    Pool_size_PSI_electrons= create_uniform_distribution(1,4, 10000)
    
    concentration_PSII_uncertainty = PS_per_chl_uncertainty * (1 - (PSI_PSII_ratio / (1 + PSI_PSII_ratio))) # mol/mol Chl
    concentration_PSI_uncertainty = PS_per_chl_uncertainty - concentration_PSII_uncertainty # mol/mol Chl

    PSI_per_area_uncertainty = concentration_PSI_uncertainty * chl_content # umol m-2 s-1
    CB6F_per_area_uncertainty = concentration_CB6F_uncertainty * chl_content
    
    #calculate a Michaelis menten curve fit on the PSI rate
    global fit_result
    fit_result = curve_fit(MM_rate_calc, np.array(CB6F_data.PAR), np.array(CB6F_data['PSI rate']))
    CB6F_rate_max = fit_result[0][0]
    MM_constant = fit_result[0][1]
    
    recovery_rate_max_uncertainty = CB6F_rate_max * PSI_per_area_uncertainty * Pool_size_PSI_electrons
    recovery_rate_max_low = np.percentile(recovery_rate_max_uncertainty, 2.5)
    recovery_rate_max_med = np.percentile(recovery_rate_max_uncertainty, 50)
    recovery_rate_max_high = np.percentile(recovery_rate_max_uncertainty, 97.5)
    
    for i in range(len(CB6F_data['PSI rate'])):
        PSI_reduction_area_list = PSI_per_area_uncertainty * CB6F_data.at[i, 'PSI rate'] * Pool_size_PSI_electrons
        CB6F_data.at[i, 'PSI reduction area low'] = np.percentile(PSI_reduction_area_list, 2.5)
        CB6F_data.at[i, 'PSI reduction area'] = np.percentile(PSI_reduction_area_list, 50)
        CB6F_data.at[i, 'PSI reduction area high'] = np.percentile(PSI_reduction_area_list, 97.5)
        
        rate_CB6F_list = PSI_reduction_area_list / CB6F_per_area_uncertainty 
        CB6F_data.at[i, 'rate CB6F low'] = np.percentile(rate_CB6F_list, 2.5)
        CB6F_data.at[i, 'rate CB6F'] = np.percentile(rate_CB6F_list, 50)
        CB6F_data.at[i, 'rate CB6F high'] = np.percentile(rate_CB6F_list, 97.5)
    
    
    
    
    
    fig, ax = plt.subplots(1,3, figsize=(12, 4))
    
    ax[0].plot(CB6F_data['PAR'], CB6F_data['PSI rate'], marker='o', ls='')
    ax[0].plot(np.arange(2000), MM_rate_calc(np.arange(2000), CB6F_rate_max, MM_constant), ls='-', marker='')
    ax[0].set_xlabel('PAR intensity')
    ax[0].set_ylabel('e- per PSI per s')
    
    
    ax[1].errorbar(CB6F_data['PAR'], CB6F_data['PSI reduction area'],
                   yerr=[CB6F_data['PSI reduction area']-CB6F_data['PSI reduction area low'],
                         CB6F_data['PSI reduction area high']-CB6F_data['PSI reduction area']],
                   ls='')
    PSI, = ax[1].plot(CB6F_data['PAR'], CB6F_data['PSI reduction area'], marker='o', ls='')
    theoretical, = ax[1].plot(np.arange(2000 ), np.arange(2000)/2)

    max_fit,  = ax[1].plot(np.arange(2000), [recovery_rate_max_med]*2000, ls='-', marker='', color='lightcoral')
    ax[1].fill_between(np.arange(2000), recovery_rate_max_low, recovery_rate_max_high, color='lightcoral', alpha=0.5)
    ax[1].set_xlabel(r'PAR intensity ($\mu mol$ $m^{-2}$ $s^{-1}$)')
    ax[1].set_ylabel(r'$\mu mol$ $e^-$ $m^{-2}$ $s^{-1}$')
    ax[1].legend([PSI, theoretical, max_fit], ['electron flow PSI', 'theoretical max (PAR/2)', 'fitted max rate'])
    ax[1].set_ylim(0,150)
    ax[1].set_xlim(0,2000)
    
    ax[2].errorbar(CB6F_data['PAR'], CB6F_data['rate CB6F'],
                   yerr=[CB6F_data['rate CB6F']-CB6F_data['rate CB6F low'],
                         CB6F_data['rate CB6F high']-CB6F_data['rate CB6F']],
                   ls='')
    ax[2].plot(CB6F_data['PAR'], CB6F_data['rate CB6F'], marker='o', ls='')
    ax[2].set_xlabel('PAR intensity ($\mu mol$ $m^{-2}$ $s^{-1}$)')
    ax[2].set_ylabel(r'e$^-$ $CB6F^{-1}$ $s^{-1}$')
    ax[2].set_ylim(0,240)
    ax[2].set_xlim(0,2000)
    

    save_name = os.path.join(f'analysis_results/{plantID}/CB6F_curve')
    fig.savefig(save_name,dpi=400)
    plt.close()     


    fig, ax = plt.subplots(1,1, figsize=(6,5))
  
    ax.errorbar(CB6F_data['PAR'], CB6F_data['PSI reduction area'],
                   yerr=[CB6F_data['PSI reduction area']-CB6F_data['PSI reduction area low'],
                         CB6F_data['PSI reduction area high']-CB6F_data['PSI reduction area']],
                   ls='')
    PSI, = ax.plot(CB6F_data['PAR'], CB6F_data['PSI reduction area'], marker='o', ls='')

    max_fit,  = ax.plot(np.arange(2000), [recovery_rate_max_med]*2000, ls='-', marker='', color='lightcoral')
    CI = ax.fill_between(np.arange(2000), recovery_rate_max_low, recovery_rate_max_high, color='lightcoral', alpha=0.5)
    ax.set_xlabel(r'PAR intensity [$\mu mol$ $m^{-2}$ $s^{-1}$]')
    ax.set_ylabel(r'CB6F recovery rate [$\mu mol$ $e^-$ $m^{-2}$ $s^{-1}$]')
    ax.legend([PSI, max_fit, CI], ['CB6F recovery rate',  'Fitted max rate', '95% CI max rate'],
              loc='upper left')
    ax.set_ylim(0,250)
    ax.set_xlim(0,2000)

    
    fig.tight_layout()
    save_name = os.path.join(f'analysis_results/{plantID}/CB6F_curve_for SI.png')
    fig.savefig(save_name,dpi=400)
    plt.close()    



    recovery_rate = (recovery_rate_max_uncertainty.mean(), recovery_rate_max_uncertainty.std(), 0)   
    return recovery_rate

#%% tech reps of Leaf absorbance
# plot the replicates and based on that think about how it should be included.
def calc_leaf_abs_variance():
    folder_name = 'leaf_absorbance_measurements'
    file_folder = os.path.join(base_directory, folder_name)
    file_list = [filename for filename in os.listdir(file_folder) if ('SF10' in filename) and (not filename.endswith('.png'))]
    leaf_absorbances = []
    for file_name in file_list:
        file_location = os.path.join(file_folder, file_name)
        
        with open(file_location, 'r') as file:
            # Skip the first 14 rows (header) in the ASCII file
            lines = file.readlines()[1:]
            data = np.array([list(map(lambda x: float(x.replace(',', '.')), line.split('\t'))) for line in lines])
            leaf_abs_wavelength = data.T[8]
            leaf_absorbance = data.T[11]
    
            leaf_absorbances.append(standardize_spectrum(leaf_abs_wavelength, leaf_absorbance, 'interpolation'))
    
    leaf_absorbances = np.array(leaf_absorbances)
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    for i in range(len(leaf_absorbances)):
        
        absorbance, = ax[0].plot(np.arange(380,751), leaf_absorbances[i], ls='-', marker='', lw=2,c='blue', alpha=0.5)
    ax[0].set_xlabel(r'Wavelengths (nm)')
    ax[0].set_ylabel('leaf level absorptance (-)')
    ax[0].set_xlim(380,750)
    ax[0].set_ylim(0,1.05)
    ax[0].set_title('leaf absorptance spectra')
    ax[0].legend(handles=[absorbance], loc='upper left')
    
    std, = ax[1].plot(np.arange(380,751), leaf_absorbances.std(axis=0), ls='-', marker='', lw=2,c='blue', alpha=0.5)
    ax[1].set_xlabel(r'Wavelengths (nm)')
    ax[1].set_ylabel('leaf level absorptance (-)')
    ax[1].set_xlim(380,750)
    #ax[1].set_ylim(0,1.05)
    ax[1].set_title('leaf absorptance std')
    ax[1].legend(handles=[std], loc='upper left')
    

    fig.tight_layout()
    save_name = os.path.join(file_folder,'tech reps of leaf absorptance.png')
    fig.savefig(save_name,dpi=400)
    plt.close()

    return leaf_absorbances.std(axis=0).mean()
#%% tech reps of streak measurements. 
def calc_stds_streak():
    tech_reps = ['SF_techrep_1_1', 'SF_techrep_1_2', 'SF_techrep_2']
    loc = 'Streak_measurements/Results'
    k_P2_list = []
    k_D2_list = []
    for measurement in tech_reps:
        k_P2,k_D2, PSI_PSII_ratio_streak, _ = analyse_streak_data(os.path.join(loc, measurement), measurement)
        k_P2_list.append(k_P2)
        k_D2_list.append(k_D2)
    k_P2_std = np.array(k_P2_list).std()
    k_D2_std = np.array(k_D2_list).std()
    return k_P2_std, k_D2_std
#%%
def analyse_priors(plant):
    global prior_table
    prior_table = pd.read_excel(os.path.join(base_directory, 'priors_of_uncertainty.xlsx'))
    prior_table['prior'] = pd.NA
    prior_table.set_index('model_parameter', inplace=True)
    parameters = prior_table.index
    for i in range(len(prior_table)):
        if prior_table.at[parameters[i],'source_mean'] == 'Literature':
            prior_table.at[parameters[i],'prior'] = (prior_table.at[parameters[i], 'mean'], 
                                                     prior_table.at[parameters[i], 'std'],
                                                     prior_table.at[parameters[i], 'lower_limit'])

    
    
    #Vcmax with Aci fit
    Vcmax_prior = (prior_table.at['Vcmax', 'mean'], prior_table.at['Vcmax', 'std'],0)
    Rd_prior = (prior_table.at['R_d', 'mean'], prior_table.at['R_d', 'std'],0)
    
    if True:
        Vcmax_prior, Rd_prior = analyse_Aci(plant.Aci_data, Vcmax_prior, Rd_prior, plant.plantID)
        prior_table.at['Vcmax', 'prior'] = Vcmax_prior

        
    #R_day with Yin method
    if True:
        Rd_prior = analyse_R_day(plant.R_day_data, Rd_prior, plant.plantID)
        prior_table.at['R_d', 'prior'] = Rd_prior
    
    #chl content and chl ab ratio
    if True:
        chl_prior = (plant.chl_tot_leaf, chl_tot_std, 0)
        prior_table.at['Chl_per_area', 'prior'] = chl_prior
        prior_table.at['Chl_a_b', 'prior'] = (plant.chl_ab_ratio, chl_ab_std, 0)

        

    if True:
        PS_ratio_prior = (0.6, 0.15, 0) 
        LHCII_per_PSII_prior = (3, 0.5, 2, 4) 
        LHCII_per_PSI_prior = (1, 0.5, 0, 3)
        
        prior_table.at['PSI_PSII_ratio', 'prior'] = PS_ratio_prior
        prior_table.at['LHCII_per_PSII', 'prior'] = LHCII_per_PSII_prior
        prior_table.at['LHCII_per_PSI', 'prior'] = LHCII_per_PSI_prior
    
    #organise fit
    if True:
        recovery_rate_prior = calc_CB6F_rate(plant.CB6F_frame, chl_prior, PS_ratio_prior, plant.plantID)
        print(recovery_rate_prior)
        prior_table.at['CB6F_recovery_rate', 'prior'] = 0
        prior_table.at['CB6F_recovery_rate', 'prior'] = recovery_rate_prior
        
    #Streak analysis
    if True:
        k_P2_prior = (plant.k_P2, k_P2_std, 0)
        k_D2_prior = (plant.k_D2, k_D2_std, 0)
        prior_table.at['K_p2', 'prior'] = k_P2_prior
        prior_table.at['K_D2', 'prior'] = k_D2_prior
        

    if True:
        if plant.k_N2 > 0:
            k_N2_prior = (plant.k_N2, 0.03, 0)
        else:
            k_N2_prior = (0.01, 0.03, 0)
        prior_table.at['K_N2', 'prior'] = k_N2_prior
    
    if True:
        leaf_abs_prior = (0, leaf_abs_std, 0)
        prior_table.at['alpha leaf', 'prior'] = leaf_abs_prior
    plant.prior_table = prior_table
    return plant


#%% Starts analysis with the data
if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle as pkl
    from scipy.optimize import curve_fit
    import math

    import pymc as pm
    import arviz as az
    import xarray as xr
    
    from initial_data_processing import plant_instance
    from initial_data_processing import find_nearest
    from initial_data_processing import calc_chl_concentration
    from initial_data_processing import calc_circle_area
    from initial_data_processing import standardize_spectrum
    
    from Streak_data_analyse import analyse_streak_data
    os.chdir("""C:/Users/jans008/repositories/ST_data_analysis/final measurements""")
    base_directory = """C:/Users/jans008/repositories/ST_data_analysis/final measurements"""


    
    # Initialize random number generator
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    
    
    
    folder_name = 'pickles'
    file_folder = os.path.join(base_directory, folder_name)
    file_list = [filename for filename in os.listdir(file_folder)]
    
    #for testing only 1 is analysed
    #file_list = file_list[6:7]
    
    
    #Calculating some general prior parameters
    chl_tot_std, chl_ab_std = calc_stds_chlorophyll()
    k_P2_std, k_D2_std = calc_stds_streak()
    leaf_abs_std = calc_leaf_abs_variance()
    print(leaf_abs_std)
    for file in file_list:
        print(f'loaded {file[:-4]} data from pickle')
        with open(f'pickles/{file}', 'rb') as file_location:
            plant = pkl.load(file_location)
        
        plant_with_prior = analyse_priors(plant)
        with open(f"pickles_with_priors/{file}", 'wb') as file_location:

            pkl.dump(plant_with_prior, file_location)
        print(f'Plant {file[:-4]} has finished analysing')

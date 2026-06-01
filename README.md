# repository_for_ST_paper
A repository containing the data sets and analysis code for the paper explaining and validating a light quality dependent photosynthesis model.

1. 
All the data is stored in original data. It is ordered by measurement type.
'Measurement_overview.xlsx' contains a table of all measurements, important time labels and measurement notes. 
This also contains the measurement labels used throughout the rest of the data structure. 
Furthermore the following data is available:
ST_measurements_licor -> gas exchange measurements during the main ST measurements
ST_measurement_sat_flashes -> Dual-PAM measurements during the main ST measurements
incident_light_spectra -> light spectra measured during the main ST measurements
lamp_scripts -> files containing the order for how the lamp is regulated

Aci_curves -> Aci curves recorded after the main ST measurements. 
CB6F_rate_measurements -> DIRK measurements done 1 or 2 days after the ST measurements. technical replicates are also in here
chlorophyll_content_measurements -> chlorophyll content measurements done 1 or 2 days after the ST measurements. technical replicates are also in here
leaf_absorbance_measurements -> leaf absorptance measurements done 1 or 2 days after the ST measurements. technical replicates are also in here
Light spectra growth chamber -> measurements of the light spectra in the growth chamber. The analysis file is also in here
light_spread_chamber -> The distribution of light over hte leaf chamber is in here. The analysis file is also in here. 
ST_validation_alternatingFR_in_Dual-PAM -> data for the measurement with constant WL and alternating FR for validating the effect of ST versus NPQ.
ST_validation_continuous in Licor -> data for the measurement with light as with the main measurement, but with continuous fluorescence recording. 

Dual_pam_measuring_light_spectrum -> Speaks for itself
priors_of_uncertainty -> the contains an overview of all the priors used for the sensitivity analysis and the literature values of the bayesian analysis. 
  The literature sources underlying it as given in the second tab. 
RG9_spectrum -> speaks for itself
uncertainty kok_method -> this contains a small analysis of the correction factor between R_day calculation with the Kok and Yin method. 


2. 
Some of the underlying modelling spectra are added to 'model parameters FvCBwd model'.
The folder also contains some of the data extracted from Taylor et al. (2019). 

3. 
The scripts analysing everything are added to analysis scripts. 
The scripts can mostly be run as is, but some of the folder structure is different in this repository than my own computer, so some directories need to be adapted before it can run. 
The scripts should be run in a specific order to generate some of the necessary files. 

initial_data_processing analyses all the underlying data into a pickle. 
obtaining priors takes the pickle and analysis it a bit further to add a prior table for all leaves. This calls streak_data_analyse to generate Streak data dependent priors. this creates a pickle_with_prior
bayesian_analysis_st_model runs the whole bayesian analysis of the model and takes the pickle_with_priors. creates sampled pickles
bayesian fit_taylor data runs the analysis of the Taylor 2019 data. 
pop_analysis_for_paper takes all the sampled pickles and combines them together to create the figures of all the leaves combined. 
theoretical model analysis finally makes the scenario figures. 





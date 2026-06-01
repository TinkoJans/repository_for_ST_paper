# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:30:59 2026

@author: jans008
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle as pkl
import arviz as az
from initial_data_processing import plant_instance
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.gridspec as gridspec
os.chdir(os.path.dirname(os.path.realpath(__file__)))
base_directory = os.path.dirname(os.path.realpath(__file__))

#%% load pickles

plant_instances = os.listdir('output_fits_sampled')
test_plants = np.array(plant_instances)
#test_plants = np.delete(test_plants, np.where(test_plants == 'Ph9'))
plant_objects = []
prior_tables = []
for test_plant in test_plants:
    with open(f'output_fits_sampled/{test_plant}/fitted_pickle.pkl', 'rb') as file:
        plant_object = pkl.load(file)
        idata = plant_object.idata
        
        
        idata_sel = plant_object.idata_sel
        plant_objects.append(plant_object)
        prior_table = plant_object.prior_table
        prior_tables.append(prior_table)
        #prior_tables.append(plant.prior_table.at['PQ_per_chl', 'prior'][1)

color_dict = {'3': 'blue',
              '4': 'green',
              '5': 'red',
              '6': 'orange',
              '9': 'purple'
              }


n_lines = 9
cmap = mpl.colormaps['plasma']
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, n_lines))
rep_colors_dict = {}
for i, color in enumerate(colors):
    rep_colors_dict[i+1] = color


#%% make plots of the distributions
fontsize =11
fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(8,8)
ax = np.empty((2,3), dtype=object)

ax[0,0] = fig.add_subplot(gs[0:4,0:4])
ax[0,1] = fig.add_subplot(gs[0:4,4:8])
ax[0,2] = fig.add_subplot(gs[4:8,0:4])
ax[1,0] = fig.add_subplot(gs[4:6,4:8])
ax[1,1] = fig.add_subplot(gs[6:8,4:8])




#fig, ax = plt.subplots(2,3, figsize=(9,6))
color_dict = {'Ph3': 'blue',
              'Ph4': 'orange',
              'Ph5': 'green',
              'Ph6': 'red',
              'Ph9': 'purple'}

RMSE_A = []
RMSE_qT = []
RMSE_QY = []
for plant in plant_objects:
    light_changes = [0,30,46,62,78]
    max_t = light_changes[4]

    
    Time_model = np.array(plant.gas_exchange_data.minutes)
    time_points = np.where((Time_model>=0)&(Time_model<=max_t)) 
    Time_model = Time_model[time_points]
    

    index_Fm = np.where((Time_model<=62)&(Time_model>61.91667))
        
    Assimilation = np.array(plant.gas_exchange_data_no_SP.A)*1e-6
    Time = np.array(plant.gas_exchange_data_no_SP.minutes)
    
    #selecting relevant time points

    time_points = np.where((Time>=light_changes[2]) &(Time<=max_t))
    Assimilation_obs = Assimilation[time_points]
    Time_obs = Time[time_points]
    
    #Obtain a mask for extracting observed time points from modelled time points. 
    obs_mask = [time_step in Time_obs for time_step in Time_model]
    
    modelled_A = plant.idata_sel.posterior["A"].quantile(0.5,dim=["draw", "chain"]).values *1e6
    modelled_A = modelled_A[obs_mask]

    ax[0,0].plot(np.arange(2.5,4.2,0.1), np.arange(2.5,4.2,0.1), alpha=0.5, c='k')
    ax[0,0].plot(Assimilation_obs*1e6, modelled_A, alpha=0.3, ls='',
            color=color_dict[plant.plantID], marker='o')

    slope, intercept = np.polyfit(Assimilation_obs*1e6, modelled_A,1)
    x = np.arange(0,10, 0.1)
    ax[0,0].plot(x, intercept+slope*x, color=color_dict[plant.plantID])

    RMSE_A.append(root_mean_squared_error(Assimilation_obs*1e6, modelled_A))

    #repeating for qT
    Time_qT = np.array(plant.dual_pam_data.minutes)
    Time_qT_obs = Time_qT[np.where((Time_qT>=light_changes[2]) &(Time_qT<=max_t))]

        
    qT_obs_mask = np.zeros(len(Time_model), dtype=bool)
    for t in Time_qT_obs:
        idx = np.argmin(np.abs(Time_model - t))
        qT_obs_mask[idx] = True

    qT_observed = (plant.dual_pam_data.at[25, 'Fm_prime'] - plant.dual_pam_data['Fm_prime']) / plant.dual_pam_data.at[25, 'Fm_prime'] *100
    

    qT_observed = np.array(qT_observed[np.where((Time_qT>=light_changes[2]) &(Time_qT<=max_t))[0]], dtype=float)
    Fm = plant.idata_sel.posterior["Fm"].quantile(0.5, dim=["draw", "chain"]).values

    qT = 100*(Fm[index_Fm] - Fm) / Fm[index_Fm]
    qT_mod = qT[np.array(qT_obs_mask)]


    ax[0,1].plot(np.arange(-1,15,0.1), np.arange(-1,15,0.1), alpha=0.5, c='k')
    ax[0,1].plot(qT_observed, qT_mod, alpha=0.3, ls='',
            color=color_dict[plant.plantID], marker='o')
    
    slope, intercept = np.polyfit(qT_observed, qT_mod,1)
    x = np.arange(0,100, 1)
    ax[0,1].plot(x, intercept+slope*x, color=color_dict[plant.plantID])
    
    
    RMSE_qT.append(root_mean_squared_error(qT_observed, qT_mod))
    
    
    
    
    
    donor_side_limitation_observed = (plant.dual_pam_data.Ps-plant.dual_pam_data.P0)/plant.Pm
    QY_observed = (plant.dual_pam_data.Phigh-plant.dual_pam_data.Ps)/plant.Pm
    
    donor_side_limitation_observed = np.array( donor_side_limitation_observed[np.where((Time_qT>=light_changes[2]) &(Time_qT<=max_t))[0]], dtype=float)
    QY_observed = np.array(QY_observed[np.where((Time_qT>=light_changes[2]) &(Time_qT<=max_t))[0]], dtype=float)

    QY_observed = QY_observed / (donor_side_limitation_observed + QY_observed)
        
    
    
    QY_modelled = plant.idata_sel.posterior["D_P700"].quantile(0.5, dim=["draw", "chain"]).values
    QY_modelled = QY_modelled[np.array(qT_obs_mask)]

    ax[0,2].plot(np.arange(0,1.2,0.1), np.arange(0,1.2,0.1), alpha=0.5, c='k')
    ax[0,2].plot(QY_observed, QY_modelled, alpha=0.3, ls='',
            color=color_dict[plant.plantID], marker='o')
    
    slope, intercept = np.polyfit(QY_observed, QY_modelled,1)
    x = np.arange(0.55,1.1, 0.01)
    ax[0,2].plot(x, intercept+slope*x, color=color_dict[plant.plantID])
    
    RMSE_QY.append(root_mean_squared_error(QY_observed, QY_modelled))
       

ax[0,0].set_xlabel(r'Observed net assimilation $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
ax[0,0].set_ylabel(r'Modelled net assimilation $[\mu mol\ m^{-2} \ s^{-1}]$', fontsize=fontsize)
ax[0,1].set_xlabel('Observed qT (%)', fontsize=fontsize)
ax[0,1].set_ylabel('Modelled qT (%)', fontsize=fontsize)
ax[0,2].set_xlabel(r'Observed $\Phi_{P700}$ (-)', fontsize=fontsize)
ax[0,2].set_ylabel(r'Modelled $\Phi_{P700}$ (-)', fontsize=fontsize)

ax[0,0].set_xlim(2.6, 4.1)
ax[0,0].set_ylim(2.6, 4.1)
ax[0,0].set_xticks(np.arange(2.6,4.2,0.4))
ax[0,0].set_yticks(np.arange(2.6,4.2,0.4))

ax[0,1].set_xlim(-1, 12)
ax[0,1].set_ylim(-1, 12)
ax[0,1].set_xticks(np.arange(0,15,3))
ax[0,1].set_yticks(np.arange(0,15,3))

ax[0,2].set_xlim(0.5,1.1)
ax[0,2].set_ylim(0.5,1.1)



RMSE_A_mean = (np.array(RMSE_A).mean(), np.array(RMSE_A).std())
RMSE_qT_mean = (np.array(RMSE_qT).mean(), np.array(RMSE_qT).std())
RMSE_QY_mean = (np.array(RMSE_QY).mean(), np.array(RMSE_QY).std())

A_string = f'RMSE: {RMSE_A_mean[0]:.2f} ' + r'$\pm$' + f' {RMSE_A_mean[1]:.2f}'
qT_string = f'RMSE: {RMSE_qT_mean[0]:.2f} ' + r'$\pm$' + f' {RMSE_qT_mean[1]:.2f}'
QY_string = f'RMSE: {RMSE_QY_mean[0]:.2f} ' + r'$\pm$' + f' {RMSE_QY_mean[1]:.2f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=1)
ax[0,0].text(0.02, 0.98, A_string, va='top',
            fontsize=fontsize, transform=ax[0,0].transAxes, bbox=props)
ax[0,1].text(0.02, 0.98, qT_string, va='top',
            fontsize=fontsize, transform=ax[0,1].transAxes, bbox=props)
ax[0,2].text(0.02, 0.98, QY_string, va='top',
            fontsize=fontsize, transform=ax[0,2].transAxes, bbox=props)


legend_names = []
legend_patches = []
for plant_num in color_dict:
    legend_patches.append(Patch(facecolor=color_dict[plant_num], alpha=1))
    legend_names.append('Ph'+ plant_num)

rep_legend = ()
for rep in ['Ph3','Ph4', 'Ph5', 'Ph6', 'Ph9']:
    handle = mlines.Line2D([], [], color=color_dict[rep], marker='s', 
                           linestyle='None', alpha=1, markersize=8, 
                           markerfacecolor=color_dict[rep], markeredgecolor=color_dict[rep])
    rep_legend = rep_legend + (handle,)
    
ax[0,0].legend([rep_legend], ['Replicates'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3, loc='lower right',
               fontsize=fontsize)


# ax[0,0].legend(legend_patches, legend_names,
#     title='plant samples', ncols=2, loc='lower right', fontsize=10)

#__________________________________________________________________________-

mean_F = []
mean_R = []
correlations = []
evaluation_range = np.arange(0,1,0.001)
for plant in plant_objects:
    samples_F = np.exp(plant.idata_sel.posterior["forward_ST_rate"].values.flatten())*60
    mean_F.append(samples_F.mean())
    #ax[1,0].hist(samples_F, bins=np.arange(0,1.2,0.03), alpha=0.5, color=color_dict[plant.plantID])
    F_kde = gaussian_kde(samples_F)
    F_kde_sampled = F_kde.evaluate(evaluation_range)
    ax[1,0].plot(evaluation_range, F_kde_sampled, color = color_dict[plant.plantID])
    
    samples_R = np.exp(plant.idata_sel.posterior["reverse_ST_rate"].values.flatten())*60
    mean_R.append(samples_R.mean())
    #ax[1,1].hist(samples_R, bins=np.arange(0,0.5,0.01), alpha=0.5, color=color_dict[plant.plantID])
    R_kde = gaussian_kde(samples_R)
    R_kde_sampled = R_kde.evaluate(evaluation_range)
    ax[1,1].plot(evaluation_range, R_kde_sampled, color = color_dict[plant.plantID])

    # ax[1,2].plot(samples_F, samples_R, alpha=0.01,
    #                                       ls='', marker='o',
    #                                       color=color_dict[plant.plantID])
    # slope, intercept = np.polyfit(samples_F, samples_R,1)
    # x = np.arange(0,1, 0.01)

    # ax[1,2].plot(x, intercept+slope*x,color=color_dict[plant.plantID])
    # correlations.append(pearsonr(samples_F, samples_R))

mean_F = np.array(mean_F)
mean_R = np.array(mean_R)
correlations = np.array(correlations)
F_string = f'Mean: {mean_F.mean():.2f} ' + r'$\pm$' + f' {mean_F.std():.2f}'
R_string = f'Mean: {mean_R.mean():.2f} ' + r'$\pm$' + f' {mean_R.std():.2f}'
correlations_string = r'R^2'+f': {correlations.mean():.2f} ' + r'$\pm$' + f' {correlations.std():.2f}'

ax[1,0].text(0.98, 0.96, F_string, va='top', ha='right',
            fontsize=fontsize, transform=ax[1,0].transAxes, bbox=props)

ax[1,1].text(0.98, 0.96, R_string, va='top', ha='right',
            fontsize=fontsize, transform=ax[1,1].transAxes, bbox=props)

# ax[1,2].text(0.02, 0.98, correlations_string, va='top', ha='left',
#             fontsize=10, transform=ax[1,2].transAxes, bbox=props)
for i in [0,1]:
    ax[1,i].set_yticks([])
    
ax[1,0].set_xlim(0,1.2)
ax[1,1].set_xlim(0,0.5)
ax[1,0].set_ylim(bottom=0)
ax[1,1].set_ylim(bottom=0)
ax[1,0].set_xlabel(r'$k_{1\rightarrow2}$ $[min^{-1}]$', fontsize=fontsize)
ax[1,0].set_ylabel('Frequency [-]', fontsize=fontsize)
ax[1,1].set_xlabel(r'$k_{2\rightarrow1}$ $[min^{-1}]$', fontsize=fontsize)
ax[1,1].set_ylabel('Frequency [-]', fontsize=fontsize)
# ax[1,2].set_xlabel(r'$k_{1\rightarrow2}$ $[min^{-1}]$')
# ax[1,2].set_ylabel(r'$k_{2\rightarrow1}$ $[min^{-1}]$')
# ax[1,2].set_xlim(0,1)
# ax[1,2].set_ylim(0,0.5)


ax[0,0].text(-0.15, 0.96, 'A', transform=ax[0,0].transAxes, 
        size=fontsize+2, weight='bold')
ax[0,1].text(-0.15, 0.96, 'B', transform=ax[0,1].transAxes, 
        size=fontsize+2, weight='bold')
ax[0,2].text(-0.15, 0.96, 'C', transform=ax[0,2].transAxes, 
        size=fontsize+2, weight='bold')
ax[1,0].text(-0.15, 0.96, 'D', transform=ax[1,0].transAxes, 
        size=fontsize+2, weight='bold')
ax[1,1].text(-0.15, 0.96, 'E', transform=ax[1,1].transAxes, 
        size=fontsize+2, weight='bold')

fig.tight_layout()
save_name = os.path.join('population_analysis_output','overview_plot_rates.png')
fig.savefig(save_name,dpi=400)
plt.show()
plt.close() 

#%% Creates overview plot of light spread chamber thingies


main_folder_name = 'light_spread_chamber'
main_folder = os.path.join(base_directory, main_folder_name)
file_list = [file for file in os.listdir(main_folder) if ('array' in file) ]
light_spread = []
for file in file_list:
    array = np.loadtxt(os.path.join(main_folder, file), delimiter=',')
    light_spread.append(array)
light_spread = np.array(light_spread)
light_spread_mean = light_spread.mean(axis=0)

fig, ax = plt.subplots(1,1, figsize=(5,5))
im0 = ax.imshow(light_spread_mean.reshape((5,5)))

Dual_Pam_fiber = plt.Circle((2,2), 0.1, color='black')
ax.add_artist(Dual_Pam_fiber)
cbar = fig.colorbar(im0, orientation='vertical')  
cbar.ax.set_ylabel(r'PAR [$\mu mol\ m^{-2}\ s^{-1}]$')

ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel('---------------------------3 cm----------------------------')
ax.set_ylabel('---------------------------3 cm----------------------------')
std = 100 * light_spread_mean.std() / light_spread_mean.mean()
#fig.tight_layout()
save_name = os.path.join('population_analysis_output',f'mean_light_spread_std_{std:.2f}_%.png')
fig.savefig(save_name,dpi=400)
plt.show()
plt.close() 

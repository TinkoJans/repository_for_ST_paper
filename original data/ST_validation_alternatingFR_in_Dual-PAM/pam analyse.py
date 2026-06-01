# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:39:39 2026

@author: jans008
"""
def convert_to_minutes(time_string):
    #takes a time string in the form of 'hhmmss' and returs as a minute
    total = 0
    total+= int(time_string[:2])*60
    total+= int(time_string[2:4])
    total+= int(time_string[4:6])/60
    return total

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    a = np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
    
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
    
    #%%

    file_list = [filename for filename in os.listdir(base_directory) if filename.startswith('F_')]
    #file_list = file_list[:5]
    Fm_list = []
    Time_list = []
    start_time = convert_to_minutes(file_list[0][9:15])
    for i in range(len(file_list)):
        SP = pd.read_excel(file_list[i], skiprows=1, decimal='.', 
                           names=['Time', 'Fluorescence'])
        Fm = SP[(SP.Time>600) & (SP.Time<800)].Fluorescence.mean()
        Fm_list.append(Fm)
        
        time = convert_to_minutes(file_list[i][9:15])
        Time_list.append(time-start_time)
    
    
    Fm_list = Fm_list/Fm_list[0]
    
    Fm_ref = find_nearest(Time_list, 28)
    qT_list = 100 * (Fm_list[Fm_ref] - Fm_list) / Fm_list[Fm_ref]
    
    #%%
    fig, ax = plt.subplots(1,1,figsize=[6,5])
 
    ax.plot(Time_list, qT_list, marker='o', ls='')
    ax.set_ylabel('qT (%)')
    ax.set_xlim(0,44)
    ax.set_ylim(-1,8)
    


    ax.set_xlabel('Time (min)')
    banner = mpatches.Rectangle((2/3, 7.5), (15.67), 1,
                                facecolor='lightcoral',
                                edgecolor='lightcoral', linewidth=0.8)
    ax.add_patch(banner)
    ax.vlines(2/3,-10,15, ls='--', color='lightgrey', zorder=0)
 
    banner = mpatches.Rectangle((15.67, 7.5), (29.67), 1,
                                facecolor='red',
                                edgecolor='red', linewidth=0.8)
    ax.add_patch(banner)
    ax.vlines(15.67,-10,15, ls='--', color='lightgrey', zorder=0)

    banner = mpatches.Rectangle((29.67, 7.5), (43.83), 1,
                                facecolor='lightcoral',
                                edgecolor='lightcoral', linewidth=0.8)
    ax.add_patch(banner)
    ax.vlines(29.67,-10,15, ls='--', color='lightgrey', zorder=0)
 

    ax.vlines(43.83,-10,15, ls='--', color='lightgrey', zorder=0)

    ax.text(0.18,0.975, 'Red', horizontalalignment='center',
               verticalalignment='center', transform=ax.transAxes)
    ax.text(0.52,0.975, 'Red + FR', horizontalalignment='center',
               verticalalignment='center', transform=ax.transAxes)
    ax.text(0.84,0.975, 'Red', horizontalalignment='center',
               verticalalignment='center', transform=ax.transAxes)
        
        
        
    fig.tight_layout()
    save_name = os.path.join('ST transitions_Red_RedFR_SI.png')
    fig.savefig(save_name,dpi=400)
    plt.show()
    plt.close() 
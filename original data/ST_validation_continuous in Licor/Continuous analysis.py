# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:54:23 2026

@author: jans008
"""

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
    
    
    data = pd.read_csv('S_260319_124252.CSV', skiprows=2, delimiter=';',decimal=',', 
                       names=['Time', 'Fluorescence', 'spare'])
    data['Time'] = data['Time'].astype(float)
    data['Fluorescence'] = data['Fluorescence'].astype(float)
    data = data[['Time', 'Fluorescence']]

    data['rolled'] = data['Fluorescence'].rolling(50, center=False).mean()
    #%%
    fig, ax = plt.subplots(1,3,figsize=[12,4])
    ax[0].plot(data.Time/60, data.Fluorescence)
    ax[0].set_ylabel('Fluorescence (a.u.)')
    ax[0].set_xlim(0,44)
    
    ax[1].plot(data.Time/60, data.Fluorescence)
    ax[1].set_xlim(15,20)
    
    ax[2].plot(data.Time/60, data.Fluorescence)
    ax[2].set_xlim(15.3,16)
    
    for i in range(3):
        ax[i].set_xlabel('Time (min)')
        ax[i].set_ylim(-0.2,1.1)
        for j in range(0,4):
            
            banner = mpatches.Rectangle((0.5, 1.0), (15.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(0.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((15.5, 1), (30.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(15.5,-10,15, ls='--', color='lightgrey', zorder=0)
    
            banner = mpatches.Rectangle((30.5, 1.0), (45.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(30.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((45.5, 1), (60.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(45.5,-10,15, ls='--', color='lightgrey', zorder=0)
    
    ax[0].text(8,1.05, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    ax[0].text(22,1.05, 'White', horizontalalignment='center',
               verticalalignment='center')
    ax[0].text(38,1.05, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    
    fig.tight_layout()
    plt.show()
    
    #%%
    fig, ax = plt.subplots(1,3,figsize=[12,4])
    ax[0].plot(data.Time/60, data.rolled)
    ax[0].set_ylabel('Fluorescence (a.u.)')
    ax[0].set_xlim(0,44)
    
    ax[1].plot(data.Time/60, data.rolled)
    ax[1].set_xlim(14.5,20)
    ax[1].set_ylabel('Fluorescence (a.u.)')
    
    ax[2].plot(data.Time/60, data.rolled)
    ax[2].set_xlim(15.3,16)
    
    for i in range(3):
        ax[i].set_xlabel('Time (min)')
        ax[i].set_ylim(-0.2,1.1)
        for j in range(0,4):
            
            banner = mpatches.Rectangle((0.5, 1.0), (15.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(0.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((15.5, 1), (30.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(15.5,-10,15, ls='--', color='lightgrey', zorder=0)
    
            banner = mpatches.Rectangle((30.5, 1.0), (45.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(30.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((45.5, 1), (60.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax[i].add_patch(banner)
            ax[i].vlines(45.5,-10,15, ls='--', color='lightgrey', zorder=0)
    
    ax[0].text(8,1.05, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    ax[0].text(22,1.05, 'White', horizontalalignment='center',
               verticalalignment='center')
    ax[0].text(38,1.05, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    
    ax[1].text(15,1.05, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    ax[1].text(17.5,1.05, 'White', horizontalalignment='center',
               verticalalignment='center')
    
    
    fig.tight_layout()
    plt.show()
    
        #%%
    fig, ax = plt.subplots(1,1,figsize=[4,4])

    
    ax.plot(data.Time/60, data.rolled)
    ax.set_xlim(14,20)
    ax.set_ylabel('Fluorescence (a.u.)')
    

    
    for i in range(1):
        ax.set_xlabel('Time (min)')
        ax.set_ylim(0,1)
        for j in range(0,4):
            
            banner = mpatches.Rectangle((0.5, 0.9), (15.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax.add_patch(banner)
            ax.vlines(0.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((15.5, 0.9), (30.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax.add_patch(banner)
            ax.vlines(15.5,-10,15, ls='--', color='lightgrey', zorder=0)
    
            banner = mpatches.Rectangle((30.5, 0.9), (45.5), 0.1,
                                        facecolor='lightcoral',
                                        edgecolor='lightcoral', linewidth=0.8)
            ax.add_patch(banner)
            ax.vlines(30.5,-10,15, ls='--', color='lightgrey', zorder=0)
     
            banner = mpatches.Rectangle((45.5, 0.9), (60.5), 0.1,
                                        facecolor='lightgrey',
                                        edgecolor='lightgrey', linewidth=0.8)
            ax.add_patch(banner)
            ax.vlines(45.5,-10,15, ls='--', color='lightgrey', zorder=0)

    ax.text(14.8,0.95, 'White + FR', horizontalalignment='center',
               verticalalignment='center')
    ax.text(17.5,0.95, 'White', horizontalalignment='center',
               verticalalignment='center')
    
    
    fig.tight_layout()
    plt.show()
    
        
        
    
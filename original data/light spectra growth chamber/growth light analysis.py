# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:43:48 2026

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
    #%%
    file_list = [filename for filename in os.listdir(base_directory) if filename.startswith('uMOL')]
    spectra_list = []
    for i in range(len(file_list)):
        spectrum = np.loadtxt(file_list[i], skiprows=1)
        spectrum = spectrum.T
        wavelengths = spectrum[0]
        spectra_list.append(spectrum[1])
                
    spectra_list = np.array(spectra_list)
    spectrum_average = spectra_list.mean(axis=0)
    
    
    fig, ax = plt.subplots(1,1, figsize =(6,5))
    
    ax.plot(wavelengths, spectrum_average, label='Growth light')
    ax.set_xlim(380,780)
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(r'Light intensity $[\mu mol\ m^{-2}\ s^{-1}\ nm^{-1}]$')
    ax.legend()
    
    fig.tight_layout()
    save_name = os.path.join('growth_light_spectrum.png')
    fig.savefig(save_name,dpi=400)
    plt.show()
    plt.close() 
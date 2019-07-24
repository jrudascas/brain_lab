import time
import numpy as np
from projects.generalize_ising_model.external_field.core_new import generalized_ising
from os import walk
from projects.generalize_ising_model.tools.utils import to_normalize, save_results, makedir
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/data/hcp_mean/J_ij.csv'
path_output = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/data/hcp_mean/results/'

# Ising Parameters
temperature_parameters = (0.05, 5, 5)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 150  # Number of simulation after thermalization
thermalize_time = 0.3  #

J = to_normalize(np.loadtxt(path_input, delimiter=','))

ts = np.linspace(temperature_parameters[0], temperature_parameters[1], num=temperature_parameters[2])
colors = ['red', 'green', 'black', 'blue', 'purple']

f = plt.figure(figsize=(18, 10))  # plot the calculated values

simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                   temperature_parameters=temperature_parameters,
                                                                   no_simulations=no_simulations,
                                                                   thermalize_time=thermalize_time)

makedir(path_output)
save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, path_output)

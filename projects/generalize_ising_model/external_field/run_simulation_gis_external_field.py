import time
import numpy as np
from generalize_ising_model.external_field.core_external_field import generalized_ising
from generalize_ising_model.ising_utils import to_normalize, save_results, makedir
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/data/hcp_mean/J_ij.csv'
path_output = '/home/brainlab/Desktop/Rudas/Data/Ising/HCP/data/hcp_mean/results/'

external_field_file = '/home/brainlab/Desktop/Jorge_External_Field/external_field.csv'
external_field_sin_all = np.loadtxt(external_field_file, delimiter=',')

# Ising Parameters
temperature_parameters = (0.025, 3.5, 10)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 1200  # Number of simulation after thermalization
thermalize_time = 0.3  #

J = to_normalize(np.loadtxt(path_input, delimiter=','))

ts = np.linspace(temperature_parameters[0], temperature_parameters[1], num=temperature_parameters[2])
colors = ['red', 'green', 'black', 'blue', 'purple']

f = plt.figure(figsize=(18, 10))  # plot the calculated values
ts = np.linspace(temperature_parameters[0], temperature_parameters[1], num=temperature_parameters[2])
for i_ef in range(1):
    i_ef += 1
    print(i_ef)
    simulated_fc, critical_temperature, E, M, S, H, E1, M1, S1, H1 = generalized_ising(J,
                                                                       temperature_parameters=temperature_parameters,
                                                                       no_simulations=no_simulations,
                                                                       thermalize_time=thermalize_time,
                                                                       external_field=external_field_sin_all[:, i_ef])
    f.add_subplot(2, 2, 1)
    plt.plot(ts, E1, marker='o', color=colors[i_ef])
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy ", fontsize=20)
    plt.axis('tight')

    f.add_subplot(2, 2, 2)
    plt.plot(ts, abs(M1), marker='o', color=colors[i_ef])
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.axis('tight')

    f.add_subplot(2, 2, 3)
    plt.plot(ts, H1, marker='o', color=colors[i_ef])
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat", fontsize=20)
    plt.axis('tight')

    f.add_subplot(2, 2, 4)
    plt.plot(ts, S1, marker='o', color=colors[i_ef])
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.axis('tight')

plt.show()
makedir(path_output)
save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, path_output)

from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import *
import matplotlib.pyplot as plt
import time
import numpy as np

path_input = '/home/brainlab/Desktop/Rudas/Scripts/ising/hcp_84_det.csv'
fc_empirical_path = '/home/brainlab/Desktop/Rudas/Scripts/ising/hcp_84_fc.csv'

# Ising Parameters
temperature_parameters = (0.002, 3, 100)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 1200  # Number of simulation after thermalization
thermalize_time = 0.3  #

J = to_normalize(np.loadtxt(path_input, delimiter=','))
fc = np.loadtxt(fc_empirical_path, delimiter=',')

start_time = time.time()
simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                   temperature_parameters=temperature_parameters,
                                                                   n_time_points=no_simulations,
                                                                   thermalize_time=thermalize_time,
                                                                   phi_variables=False)

ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
print(critical_temperature)

similarity = []
simulated_fc = np.nan_to_num(simulated_fc)
print(simulated_fc.shape)
for i in range(simulated_fc.shape[-1]):
    similarity.append(ks_test(simulated_fc[..., i], fc))
    #similarity.append(1 / mse(simulated_fc[..., i], fc))
    #plotting.plot_matrix(simulated_fc[..., i], colorbar=True)
    #plotting.show()

f = plt.figure(figsize=(18, 10))  # plot the calculated values

f.add_subplot(2, 1, 1)
plt.scatter(ts, similarity, s=50, marker='o', color='IndianRed')
plt.axvline(x=critical_temperature)

f.add_subplot(2, 1, 2)
plt.scatter(ts, S, s=50, marker='o', color='IndianRed')
plt.axvline(x=critical_temperature)
plt.show()

correlation, r = correlation_function(simulated_fc, J)

dimensionality = dim(correlation, r, find_nearest(ts, critical_temperature))
print(dimensionality)

# to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, sub_dir_output_name)

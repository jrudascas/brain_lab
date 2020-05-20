from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import *
import matplotlib.pyplot as plt
import time
import numpy as np

#path_input = '/home/brainlab/Desktop/Rudas/Data/dwitest/HCP/parcellation2/output/workingdir/preproc/_subject_id_sub1/tractography/Jij_112.csv'
#path_input = '/home/brainlab/Desktop/Rudas/Data/dwitest/HCP/parcellation3/output/workingdir/preproc/_subject_id_sub1/tractography/Jij_48.csv'
#path_input = '/home/brainlab/Desktop/Rudas/Scripts/ising/hcp_84_det.csv'
path_input = '/home/brainlab/Desktop/Rudas/Scripts/ising/hcp_84_prob.csv'

J = to_normalize(np.loadtxt(path_input, delimiter=','))

# Ising Parameters
temperature_parameters = (0.005, 20, 50)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 1200  # Number of simulation after thermalization
thermalize_time = 0.3  #

d_simulated_fc, d_critical_temperature, d_E, d_M, d_S, d_H = generalized_ising(J,
                                                                   temperature_parameters=temperature_parameters,
                                                                   n_time_points=no_simulations,
                                                                   thermalize_time=thermalize_time,
                                                                   phi_variables=False,
                                                                   type='digital')

ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
c, r = correlation_function(d_simulated_fc, J)

print(d_S)
print(d_critical_temperature)
print(ts)
index_ct = find_nearest(ts, d_critical_temperature)
print(dim(c, r, index_ct))

#f = plt.figure(figsize=(18, 10))  # plot the calculated values

#f.add_subplot(2, 1, 1)
plt.scatter(ts, d_S, s=50, marker='o', color='IndianRed')
plt.axvline(x=d_critical_temperature)
plt.show()
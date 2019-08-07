# Make random matrices for speed testing

import scipy.io
import numpy as np
from generalize_ising_model.core import generalized_ising
import time
import matplotlib.pyplot as plt
import collections
from generalize_ising_model.ising_utils import distance_wei, to_normalize, to_save_results, makedir, to_generate_randon_graph, save_graph
from generalize_ising_model.phi_project.utils import to_estimate_tpm_from_ising_model, to_calculate_mean_phi, to_save_phi

dir_output_name = '/home/user/Desktop/phiTest/speedTest'

size = int(input('Matrix size?'))
output_path = dir_output_name + '/' + 'phi/' + str(size) +'/'
makedir(output_path)

Jij = save_graph(output_path + 'Jij_' + str(size) + '.csv',to_generate_randon_graph(size, isolate=False, weighted=True))

# Ising Parameters
temperature_parameters = (-1, 5, 50)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 100 # Number of simulation after thermalization
thermalize_time = 0.3  #

makedir(dir_output_name)

#i_l = []
#j_l = []
#for v in value:
 #   for w in value:
  #      i_l.append(v - 1)
   #     j_l.append(w - 1)

#J = 1. / D[i_l, j_l]
#J[J == np.inf] = 0
#J[J == -np.inf] = 0
#J = J / np.max(J)
#J = np.reshape(J, (len(value), len(value)))

#J = to_normalize(J)

J=Jij

start_time = time.time()
print('Fitting Generalized ising model')
simulated_fc, critical_temperature, E, M, S, H, spin_mean = generalized_ising(J,
                                                                              temperature_parameters=temperature_parameters,
                                                                              no_simulations=no_simulations,
                                                                              thermalize_time=thermalize_time,
                                                                              temperature_distribution='log')

sub_dir_output_name = dir_output_name + '/' + str(size) + '/'
makedir(dir_output_name + '/' + str(size))
to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, sub_dir_output_name)
print(time.time() - start_time)

start_time = time.time()

print('Computing Phi for: ' + str(size))
# ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])
ts = np.logspace(temperature_parameters[0], np.log10(temperature_parameters[1]), temperature_parameters[2])

phi_temperature = []

phi_sum = []
phi_sus = []
cont = 0

start_time = time.time()

for t in ts:
    # print('Temperature: ' + str(t))
    tpm, fpm = to_estimate_tpm_from_ising_model(J, t)
    phi_, phiSum, phiSus = to_calculate_mean_phi(fpm, spin_mean[:, cont], t)
    phi_temperature.append(phi_)
    phi_sum.append(phiSum)
    phi_sus.append(phiSus)

    cont += 1

to_save_phi(ts, phi_temperature, phi_sum, phi_sus, S, critical_temperature, size, output_path)

print('It takes ' ,time.time() - start_time ,'seconds for a ', size, 'matrix.')

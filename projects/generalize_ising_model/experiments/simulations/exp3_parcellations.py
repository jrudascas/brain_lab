import time
import numpy as np
import os
import gc
import pickle
from generalize_ising_model.ising_utils import to_normalize, save_results
from generalize_ising_model.core import generalized_ising
from natsort import natsorted

path_input = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/data/hcp_parcellation/'
default_Jij_name = 'J_ij.csv'

# Ising Parameters

no_temperature = 150
no_simulations = 1200  # Number of simulation after thermalization
thermalize_time = 0.3  #

dir_output_name = path_input + 'simulation'
if not os.path.exists(dir_output_name):
    os.mkdir(dir_output_name)

path_input_aux = path_input + 'data/'
for dirs in natsorted(os.listdir(path_input_aux)):
    dir_output_name_case = dir_output_name + '/' + dirs + '/'
    print(dirs)

    J = to_normalize(np.loadtxt(path_input_aux + dirs + '/' + default_Jij_name, delimiter=','))

    temperature_parameters = (0.002, 3,
                              no_temperature)  # Temperature parameters (initial tempeture, final tempeture, number of steps)

    start_time = time.time()
    simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                       temperature_parameters=temperature_parameters,
                                                                       n_time_points=no_simulations,
                                                                       thermalize_time=thermalize_time,
                                                                       phi_variables = False)
    print(time.time() - start_time)
    os.mkdir(dir_output_name_case)
    save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, dir_output_name_case)

    gc.collect()

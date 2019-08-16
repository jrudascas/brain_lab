import numpy as np
import time
from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_save_results,to_generate_random_graph,to_normalize,save_graph

sizes = [250,500]
main_path = '/home/user/Desktop/Popiel/check_ising/'
temperature_params =[(1.3,200,50),(1.8,600,50)]# Params for 5,10,25,100 (-3,4,50),(-1,8,50),(0,12,50), (1,100,50),
no_simulations = 4000
thermalize_time =0.3

for ind, size in enumerate(sizes):
    save_path = main_path + str(size) + '/'

    J = save_graph(save_path + 'Jij_' + str(size) + '.csv',
                   to_normalize(to_generate_random_graph(size, isolate=False, weighted=True),netx=True))

    # Ising Parameters
    temperature_parameters = temperature_params[ind]  # Temperature parameters (initial tempeture, final tempeture, number of steps)

    start_time = time.time()
    print('Fitting Generalized Ising model for a ', size, ' by', size, ' random matrix.')
    simulated_fc, critical_temperature, E, M, S, H = generalized_ising(J,
                                                                                  temperature_parameters=temperature_parameters,
                                                                                  n_time_points=no_simulations,
                                                                                  thermalize_time=thermalize_time,
                                                                                  temperature_distribution='log')

    to_save_results(temperature_parameters, J, E, M, S, H, simulated_fc, critical_temperature, save_path,
                    temperature_distribution='log')
    print('It took ', time.time() - start_time, 'seconds to fit the generalized ising model')


from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, makedir,save_file
from projects.phi.tools.utils import load_matrix, file_exists
from projects.phi.utils import *
import time

# Ising Parameters
temperature_parameters = (0.004, 2.5, 50)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
n_time_points = 4000 # Number of simulation after thermalization
thermalize_time = 0.2  #

main_path = '/home/brainlab/Desktop/Popiel/Ising_HCP/'

parcels = ['Aud', 'CinguloOperc', 'CinguloParietal', 'DMN', 'Dorsal', 'FrontoParietal', 'Retrosplenial', 'SMhand',
           'SMmouth', 'Ventral', 'Visual']


for parcel in parcels:
    print('Running', parcel)
    parcel_path = main_path + parcel + '/'
    #results_path = sub_path + 'results/'
    save_path = parcel_path  + '/results/avg/'

    makedir(save_path)

    Jij = to_normalize(load_matrix(parcel_path + 'Jij_avg.csv'))

    start_time = time.time()

    simulated_fc, critical_temperature, E, M, S, H, spin_mean, tc = generalized_ising(Jij,
                                                                                      temperature_parameters=temperature_parameters,
                                                                                      n_time_points=n_time_points,
                                                                                      thermalize_time=thermalize_time,
                                                                                      phi_variables=True,
                                                                                      return_tc=True)

    print('It took ', time.time() - start_time, 'seconds to fit the generalized Ising model')

    makedir(save_path)
    to_save_results(temperature_parameters, Jij, E, M, S, H, simulated_fc, critical_temperature, save_path)

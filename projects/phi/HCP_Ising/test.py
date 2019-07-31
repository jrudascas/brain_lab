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
    save_path = parcel_path  + '/results/'

    makedir(save_path)
    if not file_exists(save_path + 'phi.csv'):
        Jij = to_normalize(load_matrix(parcel_path + 'Jij_avg.csv'))

        start_time = time.time()

        simulated_fc, critical_temperature, E, M, S, H, spin_mean, tc = generalized_ising(Jij,
                                                                                          temperature_parameters=temperature_parameters,
                                                                                          n_time_points=n_time_points,
                                                                                          thermalize_time=thermalize_time,
                                                                                          phi_variables=True,
                                                                                          return_tc=True)

        print('It took ',time.time()-start_time, 'seconds to fit the generalized Ising model')

        makedir(save_path)
        to_save_results(temperature_parameters, Jij, E, M, S, H, simulated_fc, critical_temperature, save_path)

        save_file(spin_mean,save_path,'spin_mean')
        save_file(tc,save_path,'time_course')

        ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])

        phi_temperature, phi_sum, phi_sus = [], [], []
        cont = 0

        start_time = time.time()

        for t in ts:
            # print('Temperature: ' + str(t))
            tpm, fpm = to_estimate_tpm_from_ising_model(Jij, t)
            phi_, phiSum, phiSus = to_calculate_mean_phi(fpm, spin_mean[:, cont], t)
            phi_temperature.append(phi_)
            phi_sum.append(phiSum)
            phi_sus.append(phiSus)

            cont += 1

        to_save_phi(ts, phi_temperature, phi_sum, phi_sus, S, critical_temperature, parcel, save_path)

        print('It takes ', time.time() - start_time, 'seconds to compute phi for the ', parcel, 'network')
    else:
        print('Done!')

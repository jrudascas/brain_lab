from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, makedir,save_file
from projects.phi.tools.utils import load_matrix, file_exists,tpm_SbyN_2
from projects.phi.utils import *
import matplotlib.pyplot as plt
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

    Jij = to_normalize(load_matrix(parcel_path + 'Jij_avg.csv'))

    start_time = time.time()

    simulated_fc, critical_temperature, E, M, S, H, spin_mean, tc = generalized_ising(Jij,
                                                                                      temperature_parameters=temperature_parameters,
                                                                                      n_time_points=n_time_points,
                                                                                      thermalize_time=thermalize_time,
                                                                                      phi_variables=True,
                                                                                      return_tc=True)

    print('It took ', time.time() - start_time, 'seconds to fit the generalized Ising model')

    ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])

    start_time = time.time()

    tpm_list = []

    for t in ts:
        # print('Temperature: ' + str(t))
        tpm, fpm = to_estimate_tpm_from_ising_model(Jij, t)

        tpm_list.append(fpm)

    tpm_tc_T = []

    for temp in range(ts.shape[-1]):
        tpm_tc, state_total, frequency = tpm_SbyN_2(tc[..., temp])

        tpm_tc_T.append(tpm_tc)
    '''
        for div in range(len(state_total)):
            if state_total[div] != 0.0:
                tpm_tc[div, :] /= state_total[div]

        tpm_tc_T.append(tpm_tc)

    save_file(np.array(tpm_list),save_path,'tpm_ising')
    save_file(np.array(tpm_tc_T),save_path,'tpm_tc')
    save_file(np.asarray(ts),save_path,'time_series')
    save_file(np.asarray(critical_temperature),save_path,'crit_temp')

    '''

    print('It takes ', time.time() - start_time, 'seconds to compute the TPM for the ', parcel, 'network')

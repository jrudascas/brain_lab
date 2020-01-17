from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, makedir,save_file
from projects.phi.tools.utils import load_matrix, file_exists,main_tpm_branch,to_calculate_mean_phi
import time
import numpy as np
import pyphi

if __name__ == '__main__':

    default_delimiter = ','
    format = '%1.5f'

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

        save_path_phi = main_path + parcel + '/phi/'

        makedir(save_path_phi)

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

        save_file(critical_temperature,save_path_phi,'critT')
        save_file(tc,save_path_phi,'time_course')

        ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2])

        start_time = time.time()

        tpm_tc_T,spin_mean_list = [],[]

        for temp in range(ts.shape[-1]):
            tpm_tc, state_total, frequency = main_tpm_branch(tc[..., temp])

            for div in range(len(state_total)):
                if state_total[div] != 0.0:
                    tpm_tc[div, :] /= state_total[div]

            tpm_tc = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm_tc))

            tpm_tc_T.append(tpm_tc)
            spin_mean_list.append(frequency)

        save_file(np.array(tpm_tc_T),save_path_phi,'tpm_tc')

        print('It takes ', time.time() - start_time, 'seconds to compute the TPM for the ', parcel, 'network')

        start_time = time.time()

        del tpm_tc_T[0:5]

        phiList, phiSumList = [], []

        for i in range(len(tpm_tc_T)):


            phi, phi_sum = to_calculate_mean_phi(tpm_tc_T[i], spin_mean_list[i])

            phiList.append(phi)
            phiSumList.append(phi_sum)


        save_file(phiList,save_path,'phi')
        save_file(phiSumList,save_path_phi,'phiSum.csv')
        print('It takes ', time.time() - start_time, 'seconds to compute phi for the ', parcel, 'network')


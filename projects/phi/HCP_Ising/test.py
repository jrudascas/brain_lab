from projects.generalize_ising_model.external_field.core_new import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, makedir
from projects.phi.tools.utils import load_matrix

# Ising Parameters
temperature_parameters = (0.004, 4, 1000)  # Temperature parameters (initial tempeture, final tempeture, number of steps)
no_simulations = 5000  # Number of simulation after thermalization
thermalize_time = 0.2  #

main_path = '/home/user/Desktop/Popiel/Ising_HCP/'

parcels = ['Aud','CinguloOperc','CinguloParietal','DMN','Dorsal','FrontoParietal','Retrosplenial','SMhand','SMmouth','Ventral','Visual']

for i in range(20):
    sub_num = i+1
    sub_path = main_path + 'sub' + str(sub_num) + '/'
    for parcel in parcels:
        parcel_path = sub_path + parcel + '/'
        results_path = sub_path + 'results/'
        save_path =  results_path + parcel + '/'
        Jij = to_normalize(load_matrix(parcel_path+'Jij.csv'))

        simulated_fc, critical_temperature, E, M, S, H = generalized_ising(Jij,
                                                                   temperature_parameters=temperature_parameters,
                                                                   no_simulations=no_simulations,
                                                                   thermalize_time=thermalize_time)

        makedir(results_path)
        makedir(save_path)
        to_save_results(temperature_parameters, Jij, E, M, S, H, simulated_fc, critical_temperature, save_path)

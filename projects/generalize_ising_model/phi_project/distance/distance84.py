from generalize_ising_model.phi_project.distance.distance_utils import *

temperature_parameters=(-1,12,100)

ts = generate_ts(temperature_parameters)
path_emp = '/home/user/Desktop/phiTest/mean_struct_corr.mat'

empFC = load_matrix(path_emp)['mean_corr_fc']

simulated_fc, critical_temperature, E, M, S, H = generalized_ising_model(empFC, temperature_parameters)

distance = []

for i in range(simulated_fc.shape[-1]):
    #distance.append(matrix_distance(simFC[:,:,i],empFC_AUD))
    distance.append(ks_test(simulated_fc[:, :, i], empFC))

plot_distance(distance,ts,critical_temperature)
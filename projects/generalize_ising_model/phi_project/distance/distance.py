from generalize_ising_model.phi_project.distance.distance_utils import *
import collections


path_emp = '/home/user/Desktop/phiTest/mean_struct_corr.mat'
path_sim = '/home/user/Desktop/phiTest/AUD/sim_fc.npy'
path_sus = '/home/user/Desktop/phiTest/AUD/susc.csv'
path_phi = '/home/user/Desktop/SinaCode/pyPhi/Ising_Networks/Aud/met/Ising_met_1_meanPhi.npz'
path_ts = '/home/user/Desktop/phiTest/AUD/ts.csv'
path_crit = '/home/user/Desktop/phiTest/AUD/ctem.csv'

rsn_index = {'AUD': [33, 34, 29, 30, 21]}#,
           #  'DMN': [9, 25, 15, 24, 7],
           #  'ECL': [7, 18, 17, 3, 19],
            # 'ECR': [67, 66, 56, 75, 52],
          #   'SAL': [66, 83, 51, 68, 75],
            # 'SEN': [16, 21, 22, 23, 33],
           #  'VIL': [59, 55, 64, 61, 77],
            # 'VIM': [4, 20, 12, 9, 73],
            # 'VIO': [20, 10, 12, 4, 6]}

dict = collections.OrderedDict(sorted(rsn_index.items()))



#empFC = load_matrix(path_emp,emp_ext)['Corr_FMRI']
empFC = load_matrix(path_emp)['mean_corr_fc']
simFC = load_matrix(path_sim)
phi = load_matrix(path_phi)
critical_temperature = load_matrix(path_crit)
ts = load_matrix(path_ts)

D, B = distance_wei(1. / empFC)

#empFC_set = {}
#for key,value in dict.items():
 #   empFC_set[key] = get_Jij()

empFC_AUD = get_Jij(dict,D)


distance = []


for i in range(simFC.shape[-1]):
    #distance.append(matrix_distance(simFC[:,:,i],empFC_AUD))
    distance.append(ks_test(simFC[:, :, i], empFC_AUD))


plot_distance(distance,ts,critical_temperature)
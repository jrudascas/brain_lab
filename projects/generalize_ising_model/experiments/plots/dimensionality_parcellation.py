from generalize_ising_model.ising_utils import to_normalize, correlation_function, dim, find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import scipy.io

path_simulation = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/data/hcp_parcellation/simulation/parcellation_48/'
#path_simulation = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/simulation/9_biological/hcp/entity_0/'


ts = np.linspace(0.05, 3, 150)

simulated_matrix = np.load(path_simulation + 'sim_fc.npy')
J = np.loadtxt(path_simulation + 'J_ij.csv', delimiter=',')
critical_temperature = np.loadtxt(path_simulation + 'ctem.csv', delimiter=',')

c, r = correlation_function(simulated_matrix, J)

index_ct = find_nearest(ts, critical_temperature)
dimensionality = dim(c, r, index_ct)
print(dimensionality)





'''
#correlation = '/home/brainlab/Desktop/Rudas/Scripts/ising/dimentionality/wd2/prefull.mat'
subject_id = 0
matlab = scipy.io.loadmat(correlation)

for subject_id in range(matlab['Corr_all'].shape[-1]):
    simulated_matrix = matlab['Corr_all'][...,subject_id]
    J = matlab['J_count_MS_Det'][...,subject_id]
    ts = np.squeeze(matlab['temp'])
    critical_temperature = np.squeeze(matlab['tc_subs'])[subject_id]

    c, r = correlation_function(simulated_matrix, J)

    index_ct = find_nearest(ts, critical_temperature)
    dimensionality = dim(c, r, index_ct)
    print(dimensionality)


correlation = '/home/brainlab/Desktop/Rudas/Scripts/ising/hcp_112.mat'
matlab = scipy.io.loadmat(correlation)

simulated_matrix = matlab['Corr_DTI']
J = matlab['J']
ts = np.squeeze(matlab['temp'])
critical_temperature = 1.9

c, r = correlation_function(simulated_matrix, J)

index_ct = find_nearest(ts, critical_temperature)
dimensionality = dim(c, r, index_ct)
print(dimensionality)
'''
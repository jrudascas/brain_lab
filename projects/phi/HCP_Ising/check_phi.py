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
    load_path = parcel_path  + '/results/'

    critTemp = np.squeeze(load_matrix(load_path + 'ctem.csv'))
    ts = np.squeeze(load_matrix(load_path+'temps.csv'))
    S = np.squeeze(load_matrix(load_path+'susc.csv'))
    phi = np.squeeze(load_matrix(load_path+'phi.csv'))
    phiSus = np.squeeze(load_matrix(load_path + 'phiSus.csv'))
    phiSum = np.squeeze(load_matrix(load_path + 'phiSum.csv'))

    f = plt.figure(figsize=(18, 10))  # plot the calculated values

    ax1 = f.add_subplot(2, 2, 1)
    ax1.scatter(ts, S, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)

    # plt.xticks(x)
    plt.axvline(x=critTemp, linestyle='--', color='k')

    ax2 = f.add_subplot(2, 2, 2)
    ax2.scatter(ts, phi, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi", fontsize=20)
    # plt.xticks(x)

    plt.axvline(x=critTemp, linestyle='--', color='k')

    ax3 = f.add_subplot(2, 2, 3)
    ax3.scatter(ts, phiSum, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sum", fontsize=20)
    # plt.xticks(x)
    plt.axvline(x=critTemp, linestyle='--', color='k')
    # plt.xticks(x, ['0', '1', '2', '3', '4', '5'])

    ax4 = f.add_subplot(2, 2, 4)
    ax4.scatter(ts, phiSus, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sus", fontsize=20)
    # ax4.xticks(x,['0','1','2','3','4','5'])
    plt.axvline(x=critTemp, linestyle='--', color='k')

    plt.show()

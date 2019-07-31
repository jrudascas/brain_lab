from projects.generalize_ising_model.tools.utils import save_file,ks_test
from projects.phi.tools.utils import load_matrix
from projects.phi.utils import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


main_path = '/home/brainlab/Desktop/Popiel/Ising_HCP/'

parcels = ['Aud', 'CinguloOperc', 'CinguloParietal', 'DMN', 'Dorsal', 'FrontoParietal', 'Retrosplenial', 'SMhand',
           'SMmouth', 'Ventral', 'Visual']



for parcel in parcels:
    print('Running', parcel)
    parcel_path = main_path + parcel + '/'
    #results_path = sub_path + 'results/'
    tpm_path = parcel_path  + '/results/'

    tpm_ising = np.squeeze(load_matrix(tpm_path+'tpm_ising.npy'))
    tpm_tc = np.squeeze(load_matrix(tpm_path+'tpm_tc.npy'))
    ts = np.squeeze(load_matrix(tpm_path + 'time_series.csv'))
    crit_temp = np.squeeze(load_matrix(tpm_path + 'crit_temp.csv'))

    ks_temp = []

    for temp in range(tpm_ising.shape[0]):
        ks_temp.append(ks_test(tpm_ising[temp,...],tpm_tc[temp,...]))

    t_star = np.where(np.asarray(ks_temp) == np.max(ks_temp))[0]


    #save_file(np.asarray(ks_temp),tpm_path,'ks_results_'+parcel)

    plt.figure()
    plt.plot(ks_temp,marker='o',color='IndianRed')
    plt.axvline(np.where(ts==crit_temp),ymax=20,ymin=-20,color='k')
    plt.axvline(t_star,color='purple')

    plt.show()




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from projects.qm_brain.utils.utils import *


isc_name = 'isc_electrodes.npy'

isc_prob_name = 'isc_electrodes_prob.npy'

avg_isc_name = ''

main_path = '/home/user/Desktop/QMBrain/BYD/'

condition_list = ['Cond10/', 'Cond12/']


occipital_R_ind = [61 ,56 ,57 ,62 ,66 ,63 ,67 ,70]
occipital_C_ind = [54 ,55]
occipital_L_ind = [52, 53, 51, 50, 49, 46, 45, 44]

parietal_R_ind = [58 ,59, 60 ,65 ,64 ,68 ,69 ,77 ,72 ,71 ,74 ,75 ,76 ,81 ,80 ,85]
parietal_C_ind = [48, 42]
parietal_L_ind = [47 ,41 ,24 ,29 ,40 ,39 ,33 ,28 ,36 ,38 ,37 ,35 ,32 ,31 ,34 ,30]

posteriorf_R_ind = [79 ,78 ,84 ,3 ,88 ,83 ,82 ,87 ,91 ,86]
posteriorf_C_ind = [4]
posteriorf_L_ind = [5 ,23 ,10 ,9 ,15 ,22 ,27 ,21 ,18 ,26]

anteriorf_R_ind = [2 ,7 ,6 ,1 ,90 ,0 ,89]
anteriorf_C_ind = [8 ,12 ,11]
anteriorf_L_ind = [14 ,13 ,16 ,17 ,20 ,19 ,25]



for condition in condition_list:

    data_list = []

    load_path = main_path + condition + '/'

    isc_results = np.squeeze(load_matrix(load_path+isc_name))
    isc_results_prob = np.squeeze(load_matrix(load_path+isc_prob_name))


    for electrode in occipital_R_ind:

        fig = plt.figure()

        # results are subject, subject, electrode

        fig.add_subplot(1,2,1)
        plt.imshow(isc_results[:,:,electrode])
        plt.colorbar()
        plt.title('Raw Data')

        fig.add_subplot(1, 2,2)
        plt.imshow(isc_results_prob[:, :, electrode])
        plt.colorbar()
        plt.title('Probability Map')

        plt.show()

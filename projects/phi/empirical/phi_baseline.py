import numpy as np
import pyphi
from projects.phi.tools.utils import *


brain_states = ['Awake','Deep','Mild','Recovery']

num_perm = 100
num_nodes = 5

main_path = '/home/user/Desktop/data_phi/Propofol/Baseline/'


for state in brain_states:

    load_path = main_path + state + '/data/'
    save_path = main_path + state + '/phi/'

    makedir2(save_path)

    for i in range(num_perm):
        freq_path = load_path + 'freq_' + str(i) + '.csv'
        tpm_path = load_path + 'tpm_' + str(i) + '.csv'
        freq = np.squeeze(load_matrix(freq_path))
        tpm_SbyS = np.squeeze(load_matrix(tpm_path))
        tpm = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm_SbyS))

        if not file_exists(save_path + 'phi_' + str(i) + '.csv'):

            phi, phi_sum = to_calculate_mean_phi(tpm, freq)
            to_save_phi(phi, phi_sum, i, save_path)

        else:
            print('The file: ', save_path + 'phi_' + str(i) + '.csv', 'already exists!')



import numpy as np
import pyphi
from projects.phi.tools.utils import *


brain_states = ['Awake','Deep','Mild','Recovery']

networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']
main_path = '/home/user/Desktop/data_phi/phi/'

for state in brain_states:

    test_path_save = main_path + state + '/SbyS/'

    makedir2(test_path_save)

    new_path_list = ['/home/user/Desktop/data_phi/Propofol/' + state + '/Auditory_parcellation_5/SbyS/data/',
                     '/home/user/Desktop/data_phi/Propofol/' + state + '/Default_parcellation_5/SbyS/data/',
                     '/home/user/Desktop/data_phi/Propofol/' + state + '/DorsalAttn_parcellation_5/SbyS/data/',
                     '/home/user/Desktop/data_phi/Propofol/' + state + '/VentralAttn_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/CinguloOperc_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/FrontoParietal_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/RetrosplenialTemporal_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMhand_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMmouth_parcellation_5/SbyS/data/',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/Visual_parcellation_5/SbyS/data/']


    tpmList, freqList = [], []


    for j in range(len(networks)):
        fold = new_path_list[j]
        network = networks[j]
        for i in range(17):
            sub_num = i + 1
            freq_path = fold + 'freq_' + str(sub_num) + '.csv'
            tpm_path = fold + 'tpm_' + str(sub_num) + '.csv'
            freq = np.squeeze(load_matrix(freq_path))
            tpm_SbyS = np.squeeze(load_matrix(tpm_path))
            tpm = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm_SbyS))
            freqList.append(freq)
            tpmList.append(tpm)

        save_list(freqList, test_path_save, state, network, type='freq')
        save_list(tpmList, test_path_save, state, network, type='tpm')
        tpmList, freqList = [], []

for network in networks:

    tpm_awake = load_matrix('/home/user/Desktop/data_phi/phi/Awake/SbyS/' + network + '/Awaketpm.npy')
    tpm_mild = load_matrix('/home/user/Desktop/data_phi/phi/Mild/SbyS/' + network + '/Mildtpm.npy')
    tpm_deep = load_matrix('/home/user/Desktop/data_phi/phi/Deep/SbyS/' + network + '/Deeptpm.npy')
    tpm_recovery = load_matrix('/home/user/Desktop/data_phi/phi/Recovery/SbyS/' + network + '/Recoverytpm.npy')

    freq_awake = load_matrix('/home/user/Desktop/data_phi/phi/Awake/SbyS/' + network + '/Awakefreq.csv')
    freq_mild = load_matrix('/home/user/Desktop/data_phi/phi/Mild/SbyS/' + network + '/Mildfreq.csv')
    freq_deep = load_matrix('/home/user/Desktop/data_phi/phi/Deep/SbyS/' + network + '/Deepfreq.csv')
    freq_recovery = load_matrix('/home/user/Desktop/data_phi/phi/Recovery/SbyS/' + network + '/Recoveryfreq.csv')

    save_path_phi = main_path + network + '/SbyS2/'

    makedir2(save_path_phi)

    for i in range(len(tpm_awake)):

        tpm_list = [tpm_awake[i, ...], tpm_mild[i, ...], tpm_deep[i, ...], tpm_recovery[i, ...]]

        spin_mean_list = [freq_awake[i, :], freq_mild[i, :], freq_deep[i, :], freq_recovery[i, :]]

        phiList, phiSumList = [], []

        if not file_exists(save_path_phi + 'phi_' + str(i) + '.csv'):

            for j in range(len(tpm_list)):

                phi, phi_sum = to_calculate_mean_phi(tpm_list[j], spin_mean_list[j])

                phiList.append(phi)
                phiSumList.append(phi_sum)

        else:
            print('The file: ', save_path_phi + 'phi_' + str(i) + '.csv', 'already exists!')

        to_save_phi(phiList, phiSumList, i, save_path_phi)
        phiList, phiSumList = [], []

    tpm_list, spin_mean_list = [],[]

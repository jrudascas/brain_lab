import numpy as np
from projects.phi.tools.utils import *


brain_states = ['Awake','Deep','Mild','Recovery']

networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']
main_path = '/home/user/Desktop/data_phi/phi/'

for state in brain_states:

    test_path_save = main_path + state + '/SbyS/'


    new_path_list = [main_path + state + '/Aud/data/',
                     main_path + state + '/DMN/data/',
                     main_path + state + '/Dorsal/data/',
                     main_path + state + '/Ventral/data/',
                     main_path + state + '/Cingulo/data/',
                     main_path + state + '/Fronto/data/',
                     main_path + state + '/Retro/data/',
                     main_path + state + '/SMhand/data/',
                     main_path + state + '/SMmouth/data/',
                     main_path + state + '/Vis/data/']


    tpmList = []


    for j in range(len(networks)):
        fold = new_path_list[j]
        network = networks[j]
        for i in range(17):
            sub_num = i + 1
            tpm_path = fold + 'tpm_' + str(sub_num) + '.csv'
            tpmList.append(load_matrix(tpm_path))

        save_list(tpmList, test_path_save, state, network, type='tpm')
        tpmList = []
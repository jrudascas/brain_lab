# Auditory Network test. 13 subjects

import numpy as np
from projects.phi.tools.utils import *

default_delimiter = ','
format = '%1.5f'

path_for_baseline = '/home/user/Desktop/data_phi/Propofol/Baseline'
save_path = '/home/user/Desktop/data_phi/Propofol/'

states = ['Awake','Mild','Deep','Recovery']
state_pos = np.arange(len(states))

baseline_list = []

for state in states:
    phi_load_list = []
    baseline_state = path_for_baseline + '/' + state + '/phi/'
    for a in range(100):
        phi_load_list.append(load_matrix(baseline_state + 'phi_' + str(a) + '.csv'))

    baseline_list.append(np.array(phi_load_list))

baseline_array = np.array(baseline_list).T
np.savetxt(save_path+'baseline_phi.csv', baseline_array, delimiter=default_delimiter, fmt=format)

networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']

for idx,network in enumerate(networks):

    mainPath = '/home/user/Desktop/data_phi/phi/' + network + '/SbyS/'

    phiList,phiSumList =[],[]

    for i in range(17):
        filePathPhi = mainPath + 'phi_' + str(i) + '.csv'
        filePathSum =  mainPath + 'phiSum_' + str(i) + '.csv'

        phiList.append(load_matrix(filePathPhi))
        phiSumList.append(load_matrix(filePathSum))

    phi_array = np.asarray(phiList)
    phi_sum_array = np.asarray(phiSumList)

    np.savetxt(save_path+network+'_phi.csv', phi_array, delimiter=default_delimiter, fmt=format)


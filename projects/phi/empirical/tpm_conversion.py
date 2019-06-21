import numpy as np
import pyphi
from projects.phi.tools.utils import *


networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']


brain_states = ['Awake','Deep','Mild','Recovery']

main_path = '/home/user/Desktop/data_phi/phi/'

for network in networks:
    for state in brain_states:
        tpm = load_matrix(main_path + state + '/' + network + '/' + state + 'tpm.npy')

        for i in range(tpm.shape[0]):

            tpm_SbyS = pyphi.convert.state_by_node2state_by_state(tpm[i,...])

            tpm_save_path = main_path + state + '/' + network

            save_tpm(tpm_SbyS,tpm_save_path,i+1)


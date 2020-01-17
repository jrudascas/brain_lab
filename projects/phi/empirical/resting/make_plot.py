# Auditory Network test. 13 subjects

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from projects.phi.tools.utils import *

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

avg_awake_baseline = np.mean(baseline_array[0,:])
avg_mild_baseline = np.mean(baseline_array[1,:])
avg_deep_baseline = np.mean(baseline_array[2,:])
avg_recovery_baseline = np.mean(baseline_array[3,:])

networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']

name_for_plotting = ['Auditory', 'DMN','Dorsal','Ventral','Cingulate', 'Frontoparietal','Retrosplenial','Sensorimotor Hand','Sensorimotor Mouth','Visual']

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

    meanPhi = []
    meanPhiSum = []

    for i in range(phi_array.shape[-1]):
        meanPhi.append(np.mean(phi_array[:,i]))
        meanPhiSum.append(np.mean(phi_sum_array[:, i]))





    plt.close()
    plt.violinplot(phi_array, state_pos, showmeans=True, showextrema=True, showmedians=False)
    plt.violinplot(baseline_array,state_pos,showmeans=True)
    plt.legend
    plt.title('$\Phi$ vs Conscious State in the' + name_for_plotting[idx] +' Network')
    plt.xlabel('Brain State')
    plt.ylabel('Phi')
    plt.ylim([0,0.7])
    plt.xticks(state_pos,states)
    plt.savefig(save_path+network+'.png',dpi=600)

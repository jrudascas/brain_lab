# Auditory Network test. 13 subjects

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from projects.phi.tools.utils import *

mainPath = '/home/user/Desktop/data_phi/phi/Vis/'

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




states = ['Awake','Mild','Deep','Recovery']
state_pos = np.arange(len(states))

plt.violinplot(phi_array, state_pos, showmeans=True, showextrema=True, showmedians=False)
plt.title('Relation of Phi and Propofol Induced Unconsciousness')
plt.xlabel('Brain State')
plt.ylabel('Phi')
plt.xticks(state_pos,states)
plt.show()

plt.violinplot(phi_sum_array, state_pos, showmeans=True, showextrema=True, showmedians=False)
plt.title('Relation of Phi and Propofol Induced Unconsciousness')
plt.xlabel('Brain State')
plt.ylabel('Phi')
plt.xticks(state_pos,states)
plt.show()
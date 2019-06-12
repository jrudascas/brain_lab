from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, correlation_function, dim, \
    find_nearest
from os import walk
import numpy as np
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from networkx.utils import *

path_simulation_output = '/home/brainlab/Desktop/new_experiment/simulation/0_sizes'

sizes_ = []
dimensionality_exp = []
for simulation in natsorted(os.listdir(path_simulation_output)):

    path_simulation = path_simulation_output + '/' + simulation

    if os.path.isdir(path_simulation):
        print()
        print(simulation)
        print()

        pkl_file = open(path_simulation + '/parameters.pkl', 'rb')
        simulation_parameters = pickle.load(pkl_file)
        pkl_file.close()
        ts = np.linspace(simulation_parameters['temperature_parameters'][0],
                         simulation_parameters['temperature_parameters'][1],
                         simulation_parameters['temperature_parameters'][2])

        dimensionality_sim = []
        for entity in natsorted(os.listdir(path_simulation)):
            path_entity = path_simulation + '/' + entity + '/'

            if os.path.isdir(path_entity):
                # print(entity)

                simulated_matrix = np.load(path_entity + 'sim_fc.npy')
                J = np.loadtxt(path_entity + 'J_ij.csv', delimiter=',')
                critical_temperature = np.loadtxt(path_entity + 'ctem.csv', delimiter=',')

                c, r = correlation_function(simulated_matrix, J)

                index_ct = find_nearest(ts, critical_temperature)
                dimensionality = dim(c, r, index_ct)
                if not np.isinf(r[-1]):
                    dimensionality_sim.append(dimensionality)
        sizes_.append(J.shape[-1])
        dimensionality_exp.append(dimensionality_sim)

fig=plt.figure()
ax1=fig.add_subplot(111)
plt.violinplot(dimensionality_exp, positions=np.array(sizes_) / 10, showmeans=True, showmedians=False)
plt.xlabel("Graph Size")
plt.ylabel("Dimensionality")
plt.xticks(np.array(sizes_) / 10, list(map(str, sizes_)))
fig.savefig('Size_vs_Dimensionality.png', dpi=1200)



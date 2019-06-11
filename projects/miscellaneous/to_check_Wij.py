from natsort import natsorted
import os
import numpy as np

path = '/home/brainlab/Desktop/Rudas/Data/Ising/experiment_2/data/6_density'
default_Jij_name = 'J_ij.csv'

for folder in natsorted(os.listdir(path)):
    for subfolder in natsorted(os.listdir(path + '/' + folder)):
            J_ij = np.loadtxt(path + '/' + folder + '/' + subfolder + '/' + default_Jij_name, delimiter=',')
            for index in range(J_ij.shape[0]):
                if np.alltrue(J_ij[index, :] == 0):
                    print(path + '/' + folder + '/' + subfolder + '/')

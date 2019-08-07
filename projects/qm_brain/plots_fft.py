from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt


def get_probability(wavefun):

    amplitude = np.abs(wavefun).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    return normAmp * normAmp

def get_n_largest(array,n=92):
    ind = np.argpartition(np.abs(array), -n)[-n:]
    return array[ind]


main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

coord_stack = zip_x_y(x,y)

condition_list = ['Cond10/','Cond12/']

for condition in condition_list:

    for i in range(13):

        subject_path = main_path + condition + str(i + 1) + '/results/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathPos = subject_path + 'position_wavefunction_1d.npy'
        filepathMom = subject_path + 'momentum_wavefunction.npy'

        psi_x = np.squeeze(load_matrix(filepathPos))

        psi_p = np.squeeze(load_matrix(filepathMom))


        pos_wavefun = data_1d_to_2d(psi_x,x,y)

        # Determine the 92 best points

        psi_p_small = np.zeros(shape=(psi_p.shape[0], psi_p.shape[1]), dtype=np.complex64)

        for t in range(psi_p.shape[0]):
            psi_p_small[t, :] = get_n_largest(psi_p[t, ...].flatten())

        momentum_prob = get_probability(psi_p_small.T)

        prob_deriv = prob_derivative(probability)

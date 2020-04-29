from projects.qm_brain.utils.utils import *
import numpy as np
import time

def get_probability(wavefun):

    amplitude = np.abs(wavefun).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    return normAmp * normAmp

def get_n_largest(array,n=92):
    ind = np.argpartition(np.abs(array), -n)[-n:]
    return array[ind]


main_path = '/home/user/Desktop/QMBrain/nyc_data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = norm_on_int_pi(load_matrix(filepathX))
y = norm_on_int_pi(load_matrix(filepathY))

coord_stack = zip_x_y(x,y)

condition_list = ['Cond10/','Cond12/']



for condition in condition_list:

    for i in range(13):

        subject_path = main_path + condition + str(i + 1) + '/'

        saveroo = subject_path + 'results/'

        makedir2(saveroo)

        save_path = saveroo + 'norm/'

        makedir2(save_path)

        print('Running for subject ', i + 1, 'in folder ', condition)

        if not file_exists(save_path+'momentum_prob_short2.csv'):
            time0 = time.time()

            filepathData = subject_path + 'data.csv'

            data = load_matrix(filepathData)

            probability,wavefun = process_eeg_data2(data)

            data = None
            probability = None
            del data,probability

            # Here I need to discuss with Andrea about how to implement this new method
            # Do I normalize the data and then transform to 2d? How do I proceed?

            # Assuming I just convert normAmp and phase to 3d

            psi = data_1d_to_2d(wavefun,x,y)

            wavefun=None
            del wavefun

            # probability_conservation_plot(len(x),probability)

            momentum_wavefunction = fft_time_warp(coord_stack,psi)

            makedir2(save_path)

            #save_file(psi,save_path,'position_wavefunction')
            psi_p = momentum_wavefunction
            del psi,momentum_wavefunction
            #save_file(momentum_wavefunction,save_path,'momentum_wavefunction')

            psi_p_small = np.zeros(shape=(psi_p.shape[0], psi_p.shape[1]), dtype=np.complex64)

            for t in range(psi_p.shape[0]):
                psi_p_small[t, :] = get_n_largest(psi_p[t, ...].flatten())

            momentum_prob = get_probability(psi_p_small.T)

            save_file(momentum_prob,save_path,'momentum_prob_short2')
            print('Time: ',time.time()-time0)

        else:
            print('Already Done!')
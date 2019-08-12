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
        filepathData = main_path + condition + str(i + 1)  + '/data.csv'

        #filepathMomentum = subject_path+'momentum_wavefunction.npy'

        #psi_p = load_matrix(filepathMomentum)

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        psi = normAmp * np.exp(1j * phase)

        xAvg = probability @ x
        yAvg = probability @ y

        deltaxAvg = derivative(xAvg)
        deltayAvg = derivative(yAvg)

        prob_deriv = prob_derivative(probability)

        pxAvg = np.sum(prob_deriv[...] * x[:], axis=1)
        pyAvg = np.sum(prob_deriv[...] * y[:], axis=1)

        plt.figure()
        plt.plot(deltaxAvg,pxAvg)
        plt.title('X')
        plt.show()

        plt.figure()
        plt.plot(deltayAvg, pyAvg)
        plt.title('Y')
        plt.show()




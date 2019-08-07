from projects.qm_brain.utils.utils import *
import numpy as np

num_subjects = 5
main_path = '/home/user/Desktop/QMBrain/BYD/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'
filepathTimes = main_path + 'times.csv'

times = load_matrix(filepathTimes)
x = load_matrix(filepathX)
y = load_matrix(filepathY)

condition_list = ['Cond10/','Cond12/']

for condition in condition_list:

    for i in range(num_subjects):

        subject_path = main_path + condition + str(i + 1) + '/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        psi = normAmp * np.exp(1j * phase)

        momentum_wavefunction = momentum_wavefunc(psi, norm=None)

        Pamplitude = np.abs(momentum_wavefunction).T

        phaseP = np.unwrap(np.angle(momentum_wavefunction)).T

        ampPMag = np.sqrt(np.sum((Pamplitude * Pamplitude).T, axis=0))

        normPAmp = (np.asarray(Pamplitude.T) / np.asarray(ampPMag)).T

        psi_p = normPAmp * np.exp(1j * phaseP)

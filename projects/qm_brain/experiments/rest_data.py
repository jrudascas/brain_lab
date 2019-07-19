from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

num_subjects = 13
main_path = '/home/user/Desktop/QMBrain/RestData/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'


x = load_matrix(filepathX)
y = load_matrix(filepathY)

for i in range(num_subjects):

    subject_path = main_path + str(i + 1) + '/'

    if not file_exists(subject_path + 'DeltaX.csv'):

        print('Running for subject ', i + 1)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        psi = normAmp * np.exp(1j * phase)

        xAvg = probability @ x
        yAvg = probability @ y

        xSqrAvg = probability @ (x * x)
        ySqrAvg = probability @ (y * y)

        dx = np.sqrt(xSqrAvg - (xAvg * xAvg))
        dy = np.sqrt(ySqrAvg - (yAvg * yAvg))

        # probability_conservation_plot(len(x),probability)

        momentum_wavefunction = momentum_wavefunc(psi)

        momenta_phase, momenta_norm_amp, momentum_prob = momenta_prob(momentum_wavefunction)

        prob_deriv = prob_derivative(probability)

        # Mass is an arbitray (as of yet) fitting parameter so set it equal to one
        m = 1

        # Calculate the average momentum
        pxAvg = np.sum(prob_deriv[...] * x[:], axis=1)
        pyAvg = np.sum(prob_deriv[...] * y[:], axis=1)

        # Calculate the average squared momentum
        pxAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(x[:]), axis=1)
        pyAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(y[:]), axis=1)

        # Get the delta p
        dpx = m * np.sqrt(pxAvgSqr - (pxAvg * pxAvg))
        dpy = m * np.sqrt(pyAvgSqr - (pyAvg * pyAvg))

        # Find all of the uncertainty relations
        uncertainty_x = dpx * dx
        uncertainty_y = dpy * dy

        # Save the values for future analysis
        save_file(dx, subject_path, 'DeltaX')
        save_file(dy, subject_path, 'DeltaY')
        save_file(dpx, subject_path, 'DeltaPX')
        save_file(dpy, subject_path, 'DeltaPY')
        save_file(uncertainty_x, subject_path, 'DeltaXDeltaPX')
        save_file(uncertainty_y, subject_path, 'DeltaYDeltaPY')

    else:
        print('The file ', subject_path + 'DeltaX.csv', ' already exists!')

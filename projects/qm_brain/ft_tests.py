from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import time
main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

coord_stack = zip_x_y(x,y)

condition_list = ['Cond10/','Cond12/']

for condition in condition_list:

    for i in range(15):



        subject_path = main_path + condition + str(i + 1) + '/'

        save_path = subject_path + 'results/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        probability,wavefun = process_eeg_data2(data)

        del data

        # Here I need to discuss with Andrea about how to implement this new method
        # Do I normalize the data and then transform to 2d? How do I proceed?

        # Assuming I just convert normAmp and phase to 3d

        psi = data_1d_to_2d(wavefun,x,y)

        xAvg = probability @ x
        yAvg = probability @ y

        xSqrAvg = probability @ (x * x)
        ySqrAvg = probability @ (y * y)

        dx = np.sqrt(xSqrAvg - (xAvg * xAvg))
        dy = np.sqrt(ySqrAvg - (yAvg * yAvg))

        # probability_conservation_plot(len(x),probability)

        start = time.time()

        momentum_wavefunction = fft_time_warp(coord_stack,psi)

        print('It took ',time.time()-start, 'seconds to compute the fft')

        momentum_wavefunction = data_2d_to_1d(momentum_wavefunction,x,y)

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

        makedir2(save_path)

        save_file(dx, save_path, 'DeltaX')
        save_file(dy, save_path, 'DeltaY')
        save_file(dpx, save_path, 'DeltaPX')
        save_file(dpy, save_path, 'DeltaPY')
        save_file(uncertainty_x, save_path, 'DeltaXDeltaPX')
        save_file(uncertainty_y, save_path, 'DeltaYDeltaPY')



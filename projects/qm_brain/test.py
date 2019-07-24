from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt




filepathChanLoc = '/home/user/Desktop/QMBrain/EEG Data/1/chanLocXY.csv'
filepathData = '/home/user/Desktop/QMBrain/EEG Data/1/data.csv'
filepathTimes = '/home/user/Desktop/QMBrain/EEG Data/1/times.csv'

data = load_matrix(filepathData)
times = load_matrix(filepathTimes)
chanLocs = load_matrix(filepathChanLoc)

phase, normAmp, probability = process_eeg_data(data)

prob2, data2 = process_eeg_data2(data)

x = chanLocs[:, 0]
y = chanLocs[:, 1]

psi = normAmp*np.exp(1j*phase)


momentum_wavefunction = momentum_wavefunc(psi)

p_wavefun2 = momentum_wavefunc(data2.T)

momenta_phase,momenta_norm_amp,momentum_prob = momenta_prob(momentum_wavefunction)

phas2,p_norm,prob_p_2 = momenta_prob(p_wavefun2)
prob_deriv = prob_derivative(probability)

p_deriv2 = prob_derivative2(probability)



print(True)

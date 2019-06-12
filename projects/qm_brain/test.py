from .utils import *
import numpy as np
import matplotlib.pyplot as plt

filepathMat = '/home/user/Desktop/QMBrain/EEG1.mat'
filepathChanLoc = '/home/user/Desktop/QMBrain/chanLocXY.csv'
filepathData = '/home/user/Desktop/QMBrain/data.csv'
filepathTimes = '/home/user/Desktop/QMBrain/times.csv'


data = load_matrix(filepathData)
times = load_matrix(filepathTimes)

chanLocs = load_matrix(filepathChanLoc)

x, y, wavefunction_position, phase, normAmp, probability = process_eeg_data(data,chanLocs)

xAvg = probability@x
yAvg = probability@y

xSqrAvg = probability@(x*x)
ySqrAvg = probability@(y*y)

dx = np.sqrt(xSqrAvg-(xAvg*xAvg))
dy = np.sqrt(ySqrAvg-(yAvg*yAvg))

#probability_conservation_plot(len(x),probability)

momentum_wavefunction = momentum_wavefunction(wavefunction_position)

momenta_phase,momenta_norm_amp,momentum_prob = momenta_prob(momentum_wavefunction)

#Alternative delta x

first_term = np.zeros((len(times),len(x)))

for j in range(len(x)):
    first_term [:,j]=probability[:,j]*np.square(x[:,j])*(1-probability[:,j])

second_term = np.zeros((len(times),len(x)))

for k in range(len(x)):
    for l in range(len(x)):
        if k>l:
            second_term[:,k] = probability[:,k]*probability[:,l]*x[:,k]*x[:,l]

delta_x = np.sqrt(first_term-2*second_term)

print(dx[0,0])
print(delta_x[0,0])
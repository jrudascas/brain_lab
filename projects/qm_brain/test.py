from utils import *
import numpy as np
import matplotlib.pyplot as plt

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

# Yay it works! Now I haveto find an efficient way of calculating this!

first_term = []
ft = []

for i in range(len(times)):
    for j in range(len(x)):
        ft.append(probability[i,j]*np.square(x[j])*(1-probability[i,j]))
    first_term.append(sum(ft))
    ft=[]


second_term = []
st = []

for i in range(len(times)):
    for k in range(len(x)):
        for l in range(len(x)):
            if k>l:
                st.append(probability[i,k]*probability[i,l]*x[k]*x[l])
    second_term.append(sum(st))
    st = []

delta_x = np.sqrt(np.array(first_term)-2*np.array(second_term))

print(dx[0])
print(delta_x[0])
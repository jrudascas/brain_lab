from utils import *
import numpy as np
import matplotlib.pyplot as plt

filepathChanLoc = '/home/user/Desktop/QMBrain/chanLocXY.csv'
filepathData = '/home/user/Desktop/QMBrain/data.csv'
filepathTimes = '/home/user/Desktop/QMBrain/times.csv'

data = load_matrix(filepathData)
times = load_matrix(filepathTimes)
chanLocs = load_matrix(filepathChanLoc)

x, y ,phase, normAmp, probability = process_eeg_data(data,chanLocs)

psi = normAmp*np.exp(1j*phase)

xAvg = probability@x
yAvg = probability@y

xSqrAvg = probability@(x*x)
ySqrAvg = probability@(y*y)

dx = np.sqrt(xSqrAvg-(xAvg*xAvg))
dy = np.sqrt(ySqrAvg-(yAvg*yAvg))

#probability_conservation_plot(len(x),probability)

momentum_wavefunction = momentum_wavefunction(psi)

momenta_phase,momenta_norm_amp,momentum_prob = momenta_prob(momentum_wavefunction)

#Alternative delta x

# Yay it works! Now I haveto find an efficient way of calculating this!

prob_deriv = prob_deriv(probability)

first_term_x,first_term_y = [],[]
ft_x, ft_y = [],[]

for i in range(len(times)):
    for j in range(len(x)):
        ft_x.append(np.square(x[j])*np.square(prob_deriv[i,j])*((1/momentum_prob[i,j])-1))
        ft_y.append(np.square(y[j]) * np.square(prob_deriv[i, j]) * ((1 / momentum_prob[i, j]) - 1))
    first_term_x.append(sum(ft_x))
    first_term_y.append(sum(ft_y))
    ft_x=[]
    ft_y=[]


second_term_x,second_term_y = [],[]
st_x,st_y = [],[]

for i in range(len(times)):
    for k in range(len(x)):
        for l in range(len(x)):
            if k>l:
                st_x.append(prob_deriv[i,k]*prob_deriv[i,l]*x[k]*x[l])
                st_y.append(prob_deriv[i, k] * prob_deriv[i, l] * y[k] * y[l])
    second_term_x.append(sum(st_x))
    second_term_y.append(sum(st_y))
    st_x = []
    st_y = []

m=1

dpx = m * np.sqrt(np.array(first_term_x)-2*np.array(second_term_x))
dpy = m * np.sqrt(np.array(first_term_y)-2*np.array(second_term_y))

uncertainty_x = dpx * dx
uncertainty_y = dpy*dy

f = plt.figure(figsize=(20,12))
plt.plot(times, uncertainty_x)
plt.title('Brain Uncertainty')
plt.xlabel('Time (microseconds)')
plt.ylabel('Uncertainty')
plt.savefig('UncertaintyX.png',dpi=600)


f = plt.figure(figsize=(20,12))
plt.plot(times, uncertainty_y)
plt.title('Brain Uncertainty')
plt.xlabel('Time (microseconds)')
plt.ylabel('Uncertainty')
plt.savefig('UncertaintyY.png',dpi=600)


path = '/home/user/Desktop/QMBrain/'

save_uncertainty(dx,path,'DeltaX')
save_uncertainty(dy,path,'DeltaY')
save_uncertainty(dpx,path,'DeltaPX')
save_uncertainty(dpy,path,'DeltaPY')
save_uncertainty(uncertainty_x,path,'DeltaXDeltaPX')
save_uncertainty(uncertainty_y,path,'DeltaYDeltaPY')
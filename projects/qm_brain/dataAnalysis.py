from projects.qm_brain.utils import *
import numpy as np
import matplotlib.pyplot as plt

filepathMat = '/home/user/Desktop/QMBrain/EEG1.mat'
filepathChanLoc = '/home/user/Desktop/QMBrain/chanLocXY.csv'
filepathData = '/home/user/Desktop/QMBrain/data.csv'
filepathTimes = '/home/user/Desktop/QMBrain/times.csv'


data = load_matrix(filepathData)
times = load_matrix(filepathTimes)

chanLocs = load_matrix(filepathChanLoc)

x, y, phase, normAmp, probability = process_eeg_data(data,chanLocs)

psi = normAmp*np.exp(1j*phase)

xAvg = probability@x
yAvg = probability@y

xSqrAvg = probability@(x*x)
ySqrAvg = probability@(y*y)

dx = np.sqrt(xSqrAvg-(xAvg*xAvg))
dy = np.sqrt(ySqrAvg-(yAvg*yAvg))

#probability_conservation_plot(len(x),probability)

momentum_wavefunction = momentum_wavefunc(psi)

momenta_phase,momenta_norm_amp,momentum_prob = momenta_prob(momentum_wavefunction)

#animation_station(xAvg,yAvg,x,y,dx,dy)

pxSum = np.zeros((len(times),len(x)),dtype=np.complex64)
pxSqrSum = np.copy(pxSum)

pySum = np.zeros((len(times),len(y)),dtype=np.complex64)
pySqrSum = np.copy(pySum)

for i in range(len(x)):
    current_x = x[i]
    current_y = y[i]

    #find closest nodes

    xDiff = current_x - x
    yDiff = current_y - y

    xPairs = np.where(abs(xDiff)>abs(yDiff))
    yPairs = np.where(abs(yDiff)>abs(xDiff))

    distance = np.sqrt(np.square(xDiff)+np.square(yDiff))

    xPairDistance = distance[xPairs]
    yPairDistance = distance[yPairs]

    firstX = sorted(xPairDistance)[0]
    secondX = sorted(xPairDistance)[1]
    firstXInd = np.where(distance==firstX)[0][0]
    secondXInd = np.where(distance==secondX)[0][0]

    firstY = sorted(yPairDistance)[0]
    secondY = sorted(yPairDistance)[1]
    firstYInd = np.where(distance == firstY)[0][0]
    secondYInd = np.where(distance == secondY)[0][0]

    diffX1 = x[firstXInd]-current_x
    diffX2 = x[secondXInd] - current_x

    diffY1 = x[firstYInd] - current_y
    diffY2 = x[secondYInd] - current_y

    psiConj = normAmp[:,i]*np.exp((-1j)*phase[:,i])
    psi_n = normAmp[:,i]*np.exp((1j)*phase[:,i])

    psi_n1x = np.squeeze(normAmp[:,firstXInd]*np.exp((1j)*(phase[:,firstXInd])))
    psi_n2x = np.squeeze(normAmp[:,secondXInd]*np.exp((1j)*(phase[:,secondXInd])))

    psi_n1y = np.squeeze(normAmp[:,firstYInd]*np.exp((1j) * (phase[:,firstYInd])))
    psi_n2y = np.squeeze(normAmp[:,secondYInd]*np.exp((1j) * (phase[:,secondYInd])))

    pxSum[:,i] = 0.5 * ((psiConj/diffX1)*(psi_n1x-psi_n)+(psiConj/diffX2)*(psi_n2x-psi_n))
    pxSqrSum[:,i] = (psiConj*((psi_n-2*psi_n1x + psi_n2x)/np.square(diffX1)))

    pySum[:,i] = 0.5 * (psiConj*((psi_n1y-psi_n)/diffY1)+psiConj*((psi_n2y-psi_n)/diffY2))
    pySqrSum[:,i] = (psiConj*((psi_n-2*psi_n1y + psi_n2y)/np.square(diffY1)))

#print(pySum)

pxAvg = np.sum(pxSum,axis=1)
pxSqrAvg = np.sum(pxSqrSum,axis=1)

pyAvg = np.sum(pySum,axis=1)
pySqrAvg = np.sum(pySqrSum,axis=1)

#Considering only the length
pxAvgL = np.abs(pxAvg)
pxSqrAvgL = np.abs(pxSqrAvg)

pyAvgL = np.abs(pyAvg)
pySqrAvgL = np.abs(pySqrAvg)

# Find uncertainties
deltaX = np.sqrt(xSqrAvg-np.square(xAvg))
deltaY = np.sqrt(ySqrAvg-np.square(yAvg))

deltaPX = np.sqrt(pxSqrAvgL-np.square(pxAvgL))
deltaPY = np.sqrt(pySqrAvgL-np.square(pyAvgL))

plot_avg(xAvg,yAvg,times)

print(min(deltaX*deltaPX), min(deltaY*deltaPY))
print(pxAvg)

plt.show()

print('Is the momentum imaginary?: ',np.iscomplex(pxAvg))

momentum_x = momentum_from_position(xAvg)
momentum_y = momentum_from_position(yAvg)

momentum_x_sqr = momentum_from_position(xSqrAvg)
momentum_y_sqr = momentum_from_position(ySqrAvg)

dpx = np.sqrt(momentum_x_sqr-(momentum_x*momentum_x))
dpy = np.sqrt(momentum_y_sqr-(momentum_y*momentum_y))

plot_avg(momentum_x,momentum_y,times,title='Average Momentum as Function of Time',ylabel='Momentum (cm/mu s)')

#animation_station(momentum_x,momentum_y,x,y,dpx,dpy)

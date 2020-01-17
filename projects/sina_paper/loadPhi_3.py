import numpy as np
import scipy.io as sio
import pyphi
import time
from projects.sina_paper.ising import gen_reservoir
import matplotlib.pyplot as plt

network = input('Network: ')
filename = 'Ising_Phi_' + network
directory = 'reduced_networks/'
# wd = 'Ising_output/'

# mat = sio.loadmat(directory + wd + filename + '.mat')

# T = mat['temp']

# phiSum = np.load(directory + 'pyPhi/' + filename + '.npy')

loadfile = np.load(directory + 'pyPhi/' + filename + '.npz')
phiSum = loadfile['phiSum']
phiSus = loadfile['phiSus']
T2 = loadfile['T2']

phiSus2 = ((5 * T2 * phiSus + phiSum) - phiSum * phiSum) / (5 * T2)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


f0 = movingaverage(phiSum, 10)
f1 = movingaverage(phiSus2, 10)
f2 = movingaverage(phiSum * phiSus2, 10)

phiplot = f0 / np.amax(f0)
phiSusplot = f1 / np.amax(f1)
phiphiSusplot = f2 / (np.amax(f2))

plt.plot(T2, phiplot, "o", label='Phi')

plt.plot(T2, phiSusplot, "o", label='Phi Susceptibility')

plt.plot(T2, phiphiSusplot, "o", label='Phi*(Phi Susceptibility)')

plt.xlabel('Temperature')
plt.title(network + ' Phi Properties')
plt.xticks(np.arange(min(T2), max(T2), 0.5, ))
plt.minorticks_on()
plt.ylim((0, 1.1))
plt.legend()
plt.show()
import numpy as np
import scipy.io as sio
# import pyphi
import time
from projects.sina_paper.ising import gen_reservoir
import matplotlib.pyplot as plt

filename = '_meanPhi.npz'
directory = '/home/user/Desktop/Popiel/ising-iit-paper/results_data/Ising_random_met_pyPhi/Ising_random_met_'
# wd = 'Ising_output/'

# mat = sio.loadmat(directory + wd + filename + '.mat')

# T = mat['temp']

# phiSum = np.load(directory + 'pyPhi/' + filename + '.npy')

sub_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,27,29,31,32,33,34,35,36,37,
            39,40,41,42,43,44,45,47,49,51,52,53,54,55,56,57,58,59,60,63,64,67,68,69,70,71,72,74,76,
            77,80,81,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,102,106,108,109,110,111,112,
            113,114,115,116,117,118,119,120,121,122,123,125,126,128,129,130,131,132,133,134,135,136,139,
            140,141,142,143,144,145,146,147,148,150,151,153,154,155,156,157,158,159,160,161,162,163,164,
            165,166,167,168,169,170,171,173,176,177,180,181,182,183,184,186,187,188,190,191,192,194,196,198,199]

phi_list, phi_sus_list, t_list = [], [], []

for ind, num in enumerate(sub_nums):

    loadfile = np.load(directory + str(num) + filename)
    phiSum = loadfile['phiSum']
    phiSus = loadfile['phiSus']
    T2 = loadfile['T2']
    phi_list.append(phiSum)
    phi_sus_list.append(phiSus)
    t_list.append(T2)


tot_phi_sum = np.array(phi_list)
tot_phi_sus = np.array(phi_sus_list)

expectation_phi_time = np.average(tot_phi_sum,axis=0)
expectation_phi_time_squared = np.average(tot_phi_sum**2,axis=0)
expectation_phi_sus_time_squared = np.average(tot_phi_sus**2,axis=0)
expectation_phi_sus_time = np.average(tot_phi_sus,axis=0)

#sigma_J_phi =

suscept_phi = expectation_phi_time_squared - expectation_phi_time**2
suscept_phi_sus = expectation_phi_sus_time_squared - expectation_phi_sus_time**2


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

smooth_phi, smooth_sus = [],[]

for i in range(tot_phi_sum.shape[0]):
    f0 = movingaverage(tot_phi_sum[i,:], 10)
    f1 = movingaverage(tot_phi_sus[i,:] * T2, 10)
    smooth_phi.append(f0/np.amax(f0))
    smooth_sus.append(f1/np.amax(f1))

smooth_tot_phi = np.array(smooth_phi)
smooth_tot_phi_sus = np.array(smooth_sus)

avg_tot_phi = np.average(smooth_tot_phi,axis=0)

avg_tot_phi_sus = np.average(smooth_tot_phi_sus,axis=0)
std_tot_phi = np.std(smooth_tot_phi,axis=0)
std_tot_phi_sus = np.std(smooth_tot_phi_sus,axis=0)
var_tot_phi = std_tot_phi ** 2
var_tot_phi_sus = std_tot_phi_sus ** 2

plt.plot(T2,avg_tot_phi,marker='o',color='k')
plt.fill_between(T2, avg_tot_phi - var_tot_phi, avg_tot_phi + var_tot_phi,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Phi with Variance')
plt.show()

plt.plot(T2,avg_tot_phi_sus,marker='o',color='k')
plt.fill_between(T2, avg_tot_phi_sus - var_tot_phi_sus, avg_tot_phi_sus + var_tot_phi_sus,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('PhiSus with Variance')
plt.show()

plt.plot(T2,avg_tot_phi,marker='o',color='k')
plt.fill_between(T2, avg_tot_phi - std_tot_phi, avg_tot_phi + std_tot_phi,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Phi with STD')
plt.show()

plt.plot(T2,avg_tot_phi_sus,marker='o',color='k')
plt.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('PhiSus with STD')
plt.show()

# Alright I have those plots basically working, time to figure out the variance plots
plt.plot(T2,var_tot_phi,marker='o',color='k')
#plt.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
 #                color='gray', alpha=0.2)
plt.semilogx()
plt.title('VarPhi')
plt.show()

plt.plot(T2,var_tot_phi_sus,marker='o',color='k')
#plt.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
 #                color='gray', alpha=0.2)
plt.semilogx()
plt.title('VarPhiSus')
plt.show()

plt.plot(T2,std_tot_phi,marker='o',color='k')
#plt.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
 #                color='gray', alpha=0.2)
plt.semilogx()
plt.title('STDPhi')
plt.show()

plt.plot(T2,std_tot_phi_sus,marker='o',color='k')
#plt.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
 #                color='gray', alpha=0.2)
plt.semilogx()
plt.title('STDPhiSus')
plt.show()


f3 = movingaverage(expectation_phi_time, 10)
f4 = movingaverage(expectation_phi_sus_time * T2, 10)
f5 = movingaverage(expectation_phi_time * expectation_phi_sus_time, 10)

f6 = movingaverage(suscept_phi,10)
f7 = movingaverage(suscept_phi_sus,10)

sus_plot = f6/np.amax(f6)
sus_sus_plot = f7/np.amax(f7)

plt.plot(T2,sus_plot,"o",label = 'Phi')
plt.plot(T2,sus_sus_plot,"o",label = 'Phi Susceptibility')
plt.semilogx()
plt.legend()
plt.show()

'''
phiplot = f0 / np.amax(f0)
phiSusplot = f1 / np.amax(f1)
phiphiSusplot = f2 / (np.amax(f2))
'''

phiplot = f3 / np.amax(f3)
phiSusplot = f4 / np.amax(f4)
phiphiSusplot = f5 / (np.amax(f5))

# smoothed and normalized data

plt.plot(T2,phiplot,"o",label = 'Phi')
plt.plot(T2,phiSusplot,"o",label = 'Phi Susceptibility')

f0 = movingaverage(phiSum, 10)
f1 = movingaverage(phiSus * T2, 10)
f2 = movingaverage(phiSum * phiSus, 10)




# Raw data
'''
plt.plot(T2, phiSum, "o", label='Phi')
plt.plot(T2,phiSus,"o",label = 'Phi Susceptibility')
'''

plt.xlabel('Temperature')
plt.title(' Phi Properties')
plt.semilogx()
plt.minorticks_on()
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from projects.qm_brain.utils.utils import load_matrix
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


filename = '_meanPhi.npz'
directory = '/home/user/Desktop/Popiel/ising-iit-paper/results_data/'
addition = 'Ising_random_met_pyPhi/Ising_random_met_'

sub_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,27,29,31,32,33,34,35,36,37,
            39,40,41,42,43,44,45,47,49,51,52,53,54,55,56,57,58,59,60,63,64,67,68,69,70,71,72,74,76,
            77,80,81,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,102,106,108,109,110,111,112,
            113,114,115,116,117,118,119,120,121,122,123,125,126,128,129,130,131,132,133,134,135,136,139,
            140,141,142,143,144,145,146,147,148,150,151,153,154,155,156,157,158,159,160,161,162,163,164,
            165,166,167,168,169,170,171,173,176,177,180,181,182,183,184,186,187,188,190,191,192,194,196,198,199]

phi_list, phi_sus_list, t_list = [], [], []

for ind, num in enumerate(sub_nums):

    loadfile = np.load(directory + addition + str(num) + filename)
    phiSum = loadfile['phiSum']
    phiSus = loadfile['phiSus']
    T2 = loadfile['T2']
    phi_list.append(phiSum)
    phi_sus_list.append(phiSus)
    t_list.append(T2)

susc = load_matrix(directory+'sus.csv')
mag =  load_matrix(directory+'mag.csv')
energy =  load_matrix(directory+'energy.csv')
spec_heat = load_matrix(directory+'spec_heat.csv')
temp = load_matrix(directory+'temp.csv')

avg_E = load_matrix(directory+'mean_energy.csv')
std_E = load_matrix(directory+'std_energy.csv')

avg_Cv = load_matrix(directory+'mean_heat.csv')
std_Cv = load_matrix(directory+'std_heat.csv')

avg_Mag = load_matrix(directory+'mean_mag.csv')
std_Mag = load_matrix(directory+'std_mag.csv')

avg_chi = load_matrix(directory+'mean_chi.csv')
std_chi = load_matrix(directory+'std_chi.csv')

tot_phi_sum = np.array(phi_list)
tot_phi_sus = np.array(phi_sus_list)

expectation_phi_time = np.average(tot_phi_sum,axis=0)
expectation_phi_time_squared = np.average(tot_phi_sum**2,axis=0)
expectation_phi_sus_time_squared = np.average(tot_phi_sus**2,axis=0)
expectation_phi_sus_time = np.average(tot_phi_sus,axis=0)


suscept_phi = expectation_phi_time_squared - expectation_phi_time**2
suscept_phi_sus = expectation_phi_sus_time_squared - expectation_phi_sus_time**2

# The first step is to find the closest values in the temperatures used

temp_list = []
idxs = []



for k in range(T2.shape[0]):
    idx,val = find_nearest(temp,T2[k])
    idxs.append(idx)
    temp_list.append(val)

T = np.array(temp_list)
idxs = np.array(idxs)

susc = susc[:,idxs]
mag = mag[:,idxs]
energy = energy[:,idxs]
spec_heat = spec_heat[:,idxs]


smooth_sus,smooth_mag,smooth_energy,smooth_heat = [],[],[],[]

smooth_phi, smooth_phi_sus = [],[]

'''
for i in range(tot_phi_sum.shape[0]):
    f0 = movingaverage(tot_phi_sum[i,:], 10)
    f1 = movingaverage(tot_phi_sus[i,:] * T2, 10)
    f2 = movingaverage(susc[i,:], 10)
    f3 = movingaverage(mag[i,:], 10)
    f4 = movingaverage(energy[i,:],10)
    f5 = movingaverage(spec_heat[i,:],10)
    smooth_phi.append(f0/np.amax(f0))
    smooth_phi_sus.append(f1/np.amax(f1))
    smooth_sus.append(f2/np.amax(f2))
    smooth_mag.append(f3/np.amax(f3))
    smooth_energy.append(f4 / np.amax(f4))
    smooth_heat.append(f5 / np.amax(f5))
'''

for i in range(tot_phi_sum.shape[0]):
    f0 = movingaverage(tot_phi_sum[i,:], 10)
    f1 = movingaverage(tot_phi_sus[i,:] * T2, 10)
    f2 = movingaverage(susc[i,:], 10)
    f3 = movingaverage(mag[i,:], 10)
    f4 = movingaverage(energy[i,:],10)
    f5 = movingaverage(spec_heat[i,:],10)
    smooth_phi.append(f0)
    smooth_phi_sus.append(f1)
    smooth_sus.append(f2)
    smooth_mag.append(f3)
    smooth_energy.append(f4)
    smooth_heat.append(f5)

smooth_tot_phi = np.array(smooth_phi)
smooth_tot_phi_sus = np.array(smooth_phi_sus)

avg_tot_phi = np.average(smooth_tot_phi,axis=0)
avg_tot_phi_sus = np.average(smooth_tot_phi_sus,axis=0)

std_tot_phi = np.std(smooth_tot_phi,axis=0)
std_tot_phi_sus = np.std(smooth_tot_phi_sus,axis=0)

avg_tot_susc = np.average(smooth_sus,axis=0)
avg_tot_mag = np.average(smooth_mag,axis=0)
avg_tot_energy = np.average(smooth_energy,axis=0)
avg_tot_heat = np.average(smooth_heat,axis=0)

std_tot_susc = np.std(smooth_sus,axis=0)
std_tot_mag = np.std(smooth_mag,axis=0)
std_tot_energy = np.std(smooth_energy,axis=0)
std_tot_heat = np.std(smooth_heat,axis=0)


T = np.delete(T,[96,97,98,99])
T = np.delete(T,[0,1,2,3,4])

T2 = np.delete(T2,[96,97,98,99])
T2 = np.delete(T2,[0,1,2,3,4])

avg_tot_mag = np.delete(avg_tot_mag,[96,97,98,99])
avg_tot_mag = np.delete(avg_tot_mag,[0,1,2,3,4])

std_tot_mag = np.delete(std_tot_mag,[96,97,98,99])
std_tot_mag = np.delete(std_tot_mag,[0,1,2,3,4])

avg_E = np.delete(avg_E,[96,97,98,99])
avg_E = np.delete(avg_E,[0,1,2,3,4])

std_E = np.delete(std_E,[96,97,98,99])
std_E = np.delete(std_E,[0,1,2,3,4])

avg_tot_phi = np.delete(avg_tot_phi,[96,97,98,99])
avg_tot_phi = np.delete(avg_tot_phi,[0,1,2,3,4])

std_tot_phi = np.delete(std_tot_phi,[96,97,98,99])
std_tot_phi = np.delete(std_tot_phi,[0,1,2,3,4])

avg_tot_susc = np.delete(avg_tot_susc,[96,97,98,99])
avg_tot_susc = np.delete(avg_tot_susc,[0,1,2,3,4])

std_tot_susc = np.delete(std_tot_susc,[96,97,98,99])
std_tot_susc = np.delete(std_tot_susc,[0,1,2,3,4])

avg_tot_heat = np.delete(avg_tot_heat,[96,97,98,99])
avg_tot_heat = np.delete(avg_tot_heat,[0,1,2,3,4])

std_tot_heat = np.delete(std_tot_heat,[96,97,98,99])
std_tot_heat = np.delete(std_tot_heat,[0,1,2,3,4])

avg_tot_phi_sus = np.delete(avg_tot_phi_sus,[96,97,98,99])
avg_tot_phi_sus = np.delete(avg_tot_phi_sus,[0,1,2,3,4])

std_tot_phi_sus = np.delete(std_tot_phi_sus,[96,97,98,99])
std_tot_phi_sus = np.delete(std_tot_phi_sus,[0,1,2,3,4])


plt.close()
fig, axs = plt.subplots(3,2)

ax0 = axs[0,0]
ax1 = axs[1,0]
ax2 = axs[2,0]
ax3 = axs[0,1]
ax4 = axs[1,1]
ax5 = axs[2,1]

x_tickss = [0.1,0.5,1,2,3,4]


ax0.scatter(T,avg_tot_mag,color='k',marker='o')
ax0.fill_between(T, avg_tot_mag - std_tot_mag, avg_tot_mag + std_tot_mag,
                 color='gray', alpha=0.2)
ax0.semilogx()
ax0.set_xticks(x_tickss)
ax0.set_title(r'$\bar{M}$')
ax0.annotate('A',xy=(0.1,1))

ax1.scatter(T,avg_E,color='k',marker='o')
ax1.fill_between(T, avg_E - std_E, avg_E + std_E,
                 color='gray', alpha=0.2)
ax1.semilogx()
ax1.set_xticks(x_tickss)
ax1.set_title(r'$\bar{E}$')
ax1.annotate('B',xy=(0.1,-0.27))

ax2.scatter(T2,avg_tot_phi,color='k',marker='o')
ax2.fill_between(T2, avg_tot_phi - std_tot_phi, avg_tot_phi + std_tot_phi,
                 color='gray', alpha=0.2)
ax2.semilogx()
ax2.set_xticks(x_tickss)
ax2.set_title(r'$\bar{\Phi}$')
ax2.annotate('C',xy=(0.1,1))

ax3.scatter(T,avg_tot_susc,color='k',marker='o')
ax3.fill_between(T, avg_tot_susc - std_tot_susc, avg_tot_susc + std_tot_susc,
                 color='gray', alpha=0.2)
ax3.semilogx()
ax3.set_xticks(x_tickss)
ax3.set_title(r'$\bar{\chi}$')
ax3.annotate('D',xy=(0.1,1))

ax4.scatter(T,avg_tot_heat,color='k',marker='o')
ax4.fill_between(T, avg_tot_heat - std_tot_heat, avg_tot_heat + std_tot_heat,
                 color='gray', alpha=0.2)
ax4.semilogx()
ax4.set_xticks(x_tickss)
ax4.set_title(r'$\bar{C}_v$')
ax4.annotate('E',xy=(0.1,1))

ax5.scatter(T2,avg_tot_phi_sus,color='k',marker='o')
ax5.fill_between(T2, avg_tot_phi_sus - std_tot_phi_sus, avg_tot_phi_sus + std_tot_phi_sus,
                 color='gray', alpha=0.2)
ax5.semilogx()
ax5.set_xticks(x_tickss)
ax5.set_title(r'$\bar{\chi}_{\Phi}$')
ax5.annotate('F',xy=(0.1,0.97))

fig.suptitle('Order and Susceptibility',fontsize=20)
fig.set_size_inches(18.5, 10.5)
plt.savefig(directory + 'plot_notnorm.pdf', dpi=600)

plt.show()

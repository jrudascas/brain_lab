import numpy as np
import scipy.io as sio
# import pyphi
import time
from projects.sina_paper.ising import gen_reservoir
import matplotlib.pyplot as plt
from projects.qm_brain.utils.utils import load_matrix

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


T2 = np.array([0.1, 0.103777, 0.10769665, 0.11176436, 0.11598569, 0.12036647,
               0.12491271, 0.12963066, 0.13452681, 0.13960789, 0.14488087, 0.15035302,
               0.15603185, 0.16192517, 0.16804108, 0.17438799, 0.18097463, 0.18781003,
               0.19490362, 0.20226512, 0.20990467, 0.21783277, 0.22606031, 0.2345986,
               0.24345939, 0.25265485, 0.26219762, 0.27210082, 0.28237806, 0.29304348,
               0.30411172, 0.31559802, 0.32751815, 0.33988851, 0.35272609, 0.36604855,
               0.3798742, 0.39422204, 0.4091118, 0.42456395, 0.44059972, 0.45724117,
               0.47451116, 0.49243344, 0.51103264, 0.53033433, 0.55036505, 0.57115233,
               0.59272475, 0.61511195, 0.63834472, 0.66245499, 0.68747591, 0.71344186,
               0.74038855, 0.76835301, 0.7973737, 0.82749049, 0.85874479, 0.89117957,
               0.92483941, 0.95977058, 0.9960211, 1.0336408, 1.0726814, 1.11319655,
               1.15524197, 1.19887544, 1.24415695, 1.29114874, 1.33991541, 1.39052399,
               1.44304406, 1.49754781, 1.55411017, 1.61280889, 1.67372465, 1.73694121,
               1.80254545, 1.87062757, 1.94128114, 2.0146033, 2.09069483, 2.16966035,
               2.25160838, 2.3366516, 2.42490689, 2.51649559, 2.61154359, 2.71018155,
               2.81254506, 2.91877485, 3.02901693, 3.14342285, 3.26214988, 3.38536123,
               3.51322628, 3.64592078, 3.78362715, 3.92653469])



filename = '_meanPhi.npz'
directory = '/home/user/Desktop/Popiel/ising-iit-paper/results_data/'

susc = load_matrix(directory+'sus.csv')
mag =  load_matrix(directory+'mag.csv')
energy =  load_matrix(directory+'energy.csv')
spec_heat = load_matrix(directory+'spec_heat.csv')
temp = load_matrix(directory+'temp.csv')

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

for i in range(susc.shape[0]):
    f0 = movingaverage(susc[i,:], 10)
    f1 = movingaverage(mag[i,:], 10)
    f2 = movingaverage(energy[i,:],10)
    f3 = movingaverage(spec_heat[i,:],10)
    smooth_sus.append(f0/np.amax(f0))
    smooth_mag.append(f1/np.amax(f1))
    smooth_energy.append(f2 / np.amax(f2))
    smooth_heat.append(f3 / np.amax(f3))


avg_tot_susc = np.average(smooth_sus,axis=0)
avg_tot_mag = np.average(smooth_mag,axis=0)
avg_tot_energy = np.average(smooth_energy,axis=0)
avg_tot_heat = np.average(smooth_heat,axis=0)

std_tot_susc = np.std(smooth_sus,axis=0)
std_tot_mag = np.std(smooth_mag,axis=0)
std_tot_energy = np.std(smooth_energy,axis=0)
std_tot_heat = np.std(smooth_heat,axis=0)

var_tot_susc = std_tot_susc ** 2
var_tot_mag = std_tot_mag ** 2
var_tot_energy = std_tot_energy ** 2
var_tot_heat = std_tot_heat ** 2


plt.plot(T,avg_tot_mag,marker='o',color='k')
plt.fill_between(T, avg_tot_mag - var_tot_mag, avg_tot_mag + var_tot_mag,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Mag with Variance')
plt.show()

plt.plot(T,avg_tot_susc,marker='o',color='k')
plt.fill_between(T, avg_tot_susc - var_tot_susc, avg_tot_susc + var_tot_susc,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Chi with Variance')
plt.show()


plt.plot(T,avg_tot_energy,marker='o',color='k')
plt.fill_between(T, avg_tot_energy - var_tot_energy, avg_tot_energy + var_tot_energy,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Energy with Variance')
plt.show()

plt.plot(T,avg_tot_heat,marker='o',color='k')
plt.fill_between(T, avg_tot_heat - var_tot_heat, avg_tot_heat + var_tot_heat,
                 color='gray', alpha=0.2)
plt.semilogx()
plt.title('Cv with Variance')
plt.show()

# Alright I have those plots basically working, time to figure out the variance plots



plt.plot(T,var_tot_mag,marker='o',color='k')
plt.semilogx()
plt.title('VarMag')
plt.show()

plt.plot(T,var_tot_susc,marker='o',color='k')
plt.semilogx()
plt.title('VarSusc')
plt.show()

plt.plot(T,var_tot_energy,marker='o',color='k')
plt.semilogx()
plt.title('VarEnergy')
plt.show()

plt.plot(T,var_tot_heat,marker='o',color='k')
plt.semilogx()
plt.title('VarCv')
plt.show()

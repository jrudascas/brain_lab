from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

main_path = '/home/user/Desktop/QMBrain/Source/Taken/'

filepathX = main_path + 'x_source_loc.csv'
#filepathY = main_path + 'y_source_loc.csv'
#filepathZ = main_path + 'z_source_loc.csv'
#filepathTimes = main_path + 'times.csv'
filepathData = main_path + 'data.csv'

#times = load_matrix(filepathTimes)
x = load_matrix(filepathX)
#y = load_matrix(filepathY)
#z = load_matrix(filepathZ)
data = load_matrix(filepathData)

phase, normAmp, probability = process_eeg_data(data)

del data

psi = normAmp * np.exp(1j * phase)
del normAmp, phase

xAvg = probability @ x
#yAvg = probability @ y
#zAvg = probability @ z

xSqrAvg = probability @ (x * x)
#ySqrAvg = probability @ (y * y)
#zSqrAvg = probability @ (z * z)

dx = np.sqrt(xSqrAvg - (xAvg * xAvg))
#dy = np.sqrt(ySqrAvg - (yAvg * yAvg))
#dz = np.sqrt(zSqrAvg - (zAvg * zAvg))

# probability_conservation_plot(len(x),probability)

momentum_wavefunction = momentum_wavefunc(psi)

del psi

momenta_phase, momenta_norm_amp, momentum_prob = momenta_prob(momentum_wavefunction)

del momenta_phase, momenta_norm_amp

prob_deriv = prob_derivative(probability)

del probability

m = 1



pxAvg = np.sum(prob_deriv[...] * x [:], axis=1)

pxAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(x[:]),axis=1)


dpx = m * np.sqrt(pxAvgSqr-(pxAvg*pxAvg))


uncertainty_x = dpx * dx
#uncertainty_y = dpy * dy
#uncertainty_z = dpz * dz

f = plt.figure(figsize=(20, 12))
plt.plot(np.arange(uncertainty_x.shape), uncertainty_x)
plt.title('Brain Uncertainty (x)')
plt.xlabel('Time (microseconds)')
plt.ylabel('Uncertainty')
plt.savefig(main_path + 'UncertaintyX.png', dpi=600)
plt.close()

'''

f = plt.figure(figsize=(20, 12))
plt.plot(times, uncertainty_y)
plt.title('Brain Uncertainty (y)')
plt.xlabel('Time (microseconds)')
plt.ylabel('Uncertainty')
plt.savefig(main_path + 'UncertaintyY.png', dpi=600)
plt.close()

f = plt.figure(figsize=(20, 12))
plt.plot(times, uncertainty_z)
plt.title('Brain Uncertainty (z)')
plt.xlabel('Time (microseconds)')
plt.ylabel('Uncertainty')
plt.savefig(main_path + 'UncertaintyZ.png', dpi=600)
plt.close()

save_file(dx, main_path, 'DeltaX')
save_file(dy, main_path, 'DeltaY')
save_file(dz, main_path, 'DeltaZ')
save_file(dpx, main_path, 'DeltaPX')
save_file(dpy, main_path, 'DeltaPY')
save_file(dpz, main_path, 'DeltaPZ')
save_file(uncertainty_x, main_path, 'DeltaXDeltaPX')
save_file(uncertainty_y, main_path, 'DeltaYDeltaPY')
save_file(uncertainty_z, main_path, 'DeltaZDeltaPZ')

'''

save_file(uncertainty_x, main_path, 'DeltaXDeltaPX')
save_file(dx, main_path, 'DeltaX')
save_file(dpx, main_path, 'DeltaPX')

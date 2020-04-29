import numpy as np
import os
from scipy.signal import hilbert
import scipy.io

# Necessary functions

def load_matrix(filepath):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)

def file_exists(filename):
    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False

def hilberto(data):
    assert len(data.shape) == 2
    if data.shape[1] > data.shape[0]:
        return hilbert(data, axis=1)
    else:
        return hilbert(data, axis=0).T

def process_eeg_data(data):


    hilbertTransData = hilberto(data)

    amplitude = np.abs(hilbertTransData).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    probability = normAmp * normAmp

    del ampMag, amplitude

    return probability, hilbertTransData

def momentum_wavefunc(pos_wavefunc,norm='ortho',axis=1):
    momentum_wavefunction = np.fft.fft(pos_wavefunc,norm=norm,axis=axis)
    return momentum_wavefunction

def momenta_prob(momentum_wavefunction):

    pAmp = np.abs(momentum_wavefunction)

    ampMag = np.sqrt(np.sum((pAmp * pAmp).T, axis=0))

    normpAmp = (np.asarray(pAmp.T) / np.asarray(ampMag)).T

    momentum_prob = normpAmp * normpAmp

    del ampMag, pAmp,normpAmp,

    return momentum_prob

def prob_derivative(probability,axis=0):
    return np.diff(probability,axis=axis)

def save_file(data,path,name):

    default_delimiter = ','
    format = '%1.5f'

    if len(data.shape) <= 2:
        file = path + str(name) + '.csv'

        if not file_exists(file):
            np.savetxt(file, np.asarray(data), delimiter=default_delimiter, fmt=format)
    else:
        file = path + str(name) + '.npy'
        if not file_exists(file):
            np.save(file,np.asarray([data]))

# Main file path to the folder containing all of the relevant materials
main_path = '/home/user/Desktop/QMBrain/Source/Taken/'

# Filename extensions for all of the relevant datasets
filepathX = main_path + 'x_source_loc.csv'
filepathY = main_path + 'y_source_loc.csv'
filepathZ = main_path + 'z_source_loc.csv'
filepathData = main_path + 'data.csv'

#Load in the position matrices
x = load_matrix(filepathX)
y = load_matrix(filepathY)
z = load_matrix(filepathZ)

#Process the EEG data, returning the probability and the un-normalized wavefunction so it can be converted to momentum space
probability,hilbert_data = process_eeg_data(load_matrix(filepathData))

# Find the average positions
xAvg = probability @ x
yAvg = probability @ y
zAvg = probability @ z

# Find the square of the average positions
xSqrAvg = probability @ (x * x)
ySqrAvg = probability @ (y * y)
zSqrAvg = probability @ (z * z)

# Calculate delta x/y/z or the standard deviation
dx = np.sqrt(xSqrAvg - (xAvg * xAvg))
dy = np.sqrt(ySqrAvg - (yAvg * yAvg))
dz = np.sqrt(zSqrAvg - (zAvg * zAvg))

# Get the momentum wavefunction (unnormalized) by fourier transforming the unnormalized position wavefunction
momentum_wavefunction = momentum_wavefunc(hilbert_data)

del hilbert_data

# Get the momentum probability by performing the same analysis as with position wavefunction
momentum_prob = momenta_prob(momentum_wavefunction)

# Take the derivative of the probability in position space as the model requires
prob_deriv = prob_derivative(probability)

del probability

# Mass is an arbitray (as of yet) fitting parameter so set it equal to one
m = 1

# Calculate the average momentum
pxAvg = np.sum(prob_deriv[...] * x [:], axis=1)
pyAvg = np.sum(prob_deriv[...] * y [:], axis=1)
pzAvg = np.sum(prob_deriv[...] * z [:], axis=1)

# Calculate the average squared momentum
pxAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(x[:]),axis=1)
pyAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(y[:]),axis=1)
pzAvgSqr = np.sum(np.square(prob_deriv[:, :]) * ((1 / momentum_prob[:, :])) * np.square(z[:]),axis=1)

# Get the delta p
dpx = m * np.sqrt(pxAvgSqr-(pxAvg*pxAvg))
dpy = m * np.sqrt(pyAvgSqr-(pyAvg*pyAvg))
dpz = m * np.sqrt(pzAvgSqr-(pzAvg*pzAvg))

# Find all of the uncertainty relations
uncertainty_x = dpx * dx
uncertainty_y = dpy * dy
uncertainty_z = dpz * dz

# Save the values for future analysis
save_file(dx, main_path, 'DeltaX')
save_file(dy, main_path, 'DeltaY')
save_file(dz, main_path, 'DeltaZ')
save_file(dpx, main_path, 'DeltaPX')
save_file(dpy, main_path, 'DeltaPY')
save_file(dpz, main_path, 'DeltaPZ')
save_file(uncertainty_x, main_path, 'DeltaXDeltaPX')
save_file(uncertainty_y, main_path, 'DeltaYDeltaPY')
save_file(uncertainty_z, main_path, 'DeltaZDeltaPZ')


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from projects.phi.tools.utils import *

temperature_parameters = (0.004, 2.5, 50)
ts = np.linspace(temperature_parameters[0], temperature_parameters[1], temperature_parameters[2]).tolist()

del ts[0:5]

filePathPhi = '/home/brainlab/Desktop/Popiel/Ising_HCP/Aud/phi/phi.csv'
filePathSum =  '/home/brainlab/Desktop/Popiel/Ising_HCP/Aud/phi/phiSum.csv'
filepathCrit = ''
phi = load_matrix(filePathPhi)
phiSum = load_matrix(filePathSum)



plt.scatter(ts,phi)
plt.xlabel('Temperature')
plt.ylabel('Phi')
plt.show()


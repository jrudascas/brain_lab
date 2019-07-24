import numpy as np
import pyphi
from projects.phi.tools.utils import *


networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']

brain_states = ['Awake','Deep','Mild','Recovery']

main_path = '/home/user/Desktop/data_phi/phi/'


tpm = load_matrix('/home/user/Desktop/data_phi/phi/Awake/DMN/data/tpm_1.csv')
spin_mean = load_matrix('/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/data/freq_1.csv')

phi,phiSum = to_calculate_mean_phi(tpm,spin_mean)

'''
It basically works. The issue is they set epsilon to 1e-6 and upon double conversion there are a few elements
that only have 1e-5 so it is technically non-zero. Conditional independence holds and they can back off.

'''
from projects.phi.tools.utils import *

tpm_path_concat = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/data/tpm_2.csv'
freq_path_concat = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/data/freq_2.csv'

tpm_path_eps = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/eps/data/tpm_1.csv'
freq_path_eps = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/eps/data/freq_1.csv'

tpm_path_og = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/og/data/tpm_1.csv'
freq_path_og = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/og/data/freq_1.csv'


path_output_concat = '/home/user/Desktop/data_phi/phi/concat/'
path_output_eps = '/home/user/Desktop/data_phi/phi/eps/'
path_output_og = '/home/user/Desktop/data_phi/phi/og/'

tpm = load_matrix(tpm_path_concat)
spin_mean = load_matrix(freq_path_concat)

phi,phi_sum = to_calculate_mean_phi(tpm,spin_mean)

to_save_phi(phi,phi_sum,1,path_output_concat)



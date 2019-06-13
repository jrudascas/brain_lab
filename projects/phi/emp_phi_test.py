from projects.phi.tools.utils import *

tpm_path_awake = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/data/tpm_2.csv'
freq_path_awake = '/home/user/Desktop/data_phi/Propofol/Awake/Default_parcellation_5/data/freq_2.csv'

tpm_path_mild = '/home/user/Desktop/data_phi/Propofol/Mild/Default_parcellation_5/data/tpm_2.csv'
freq_path_mild = '/home/user/Desktop/data_phi/Propofol/Mild/Default_parcellation_5/data/freq_2.csv'

tpm_path_deep = '/home/user/Desktop/data_phi/Propofol/Deep/Default_parcellation_5/data/tpm_2.csv'
freq_path_deep = '/home/user/Desktop/data_phi/Propofol/Deep/Default_parcellation_5/data/freq_2.csv'

tpm_path_recovery = '/home/user/Desktop/data_phi/Propofol/Recovery/Default_parcellation_5/data/tpm_2.csv'
freq_path_recovery = '/home/user/Desktop/data_phi/Propofol/Recovery/Default_parcellation_5/data/freq_2.csv'


path_output_awake = '/home/user/Desktop/data_phi/phi/concat/Awake/'
path_output_mild = '/home/user/Desktop/data_phi/phi/concat/Mild/'
path_output_deep = '/home/user/Desktop/data_phi/phi/concat/Deep/'
path_output_recovery = '/home/user/Desktop/data_phi/phi/concat/Recovery/'

path_output_list = [path_output_awake,path_output_mild,path_output_deep,path_output_recovery]

tpm_awake = load_matrix(tpm_path_awake)
spin_mean_awake = load_matrix(freq_path_awake)

tpm_mild = load_matrix(tpm_path_mild)
spin_mean_mild = load_matrix(freq_path_mild)

tpm_deep = load_matrix(tpm_path_deep)
spin_mean_deep = load_matrix(freq_path_deep)

tpm_recovery = load_matrix(tpm_path_recovery)
spin_mean_recovery = load_matrix(freq_path_recovery)

tpm_list = [tpm_awake,tpm_mild,tpm_deep,tpm_recovery]
spin_mean_list = [spin_mean_awake,spin_mean_mild,spin_mean_deep,spin_mean_recovery]

for i in range(len(tpm_list)):
    phi,phi_sum = to_calculate_mean_phi(tpm_list[i],spin_mean_list[i])
    to_save_phi(phi,phi_sum,1,path_output_list[i])



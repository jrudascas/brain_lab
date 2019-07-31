from projects.phi.tools.utils import load_matrix,save_Jij,makedir2,make_Jij_array,avg_Jij,file_exists
import numpy as np

main_path = '/home/brainlab/Desktop/Popiel/Ising_HCP/'

parcels = ['Aud','CinguloOperc','CinguloParietal','DMN','Dorsal','FrontoParietal','Retrosplenial','SMhand','SMmouth','Ventral','Visual']

j_name = 'Jij.csv'

for i in range(20):
    sub_num = i+1
    sub_path = main_path + 'sub' + str(sub_num) + '/'
    for parcel in parcels:
        print('Running' , parcel, 'network for Subject: ',sub_num)
        parcel_path = sub_path + parcel + '/'
        save_path = main_path + parcel
        filepath = parcel_path  + j_name
        Jij = load_matrix(filepath)

        makedir2(save_path)
        save_Jij(Jij, save_path, sub_num)



new_path_list = ['/home/brainlab/Desktop/Popiel/Ising_HCP/Aud/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/CinguloOperc/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/CinguloParietal/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/DMN/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/Dorsal/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/FrontoParietal/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/Retrosplenial/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/SMhand/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/SMmouth/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/Ventral/',
                 '/home/brainlab/Desktop/Popiel/Ising_HCP/Visual/']

for fold in new_path_list:
    J = make_Jij_array(fold)
    J = avg_Jij(J)

    default_delimiter = ','
    format = '%1.5f'

    filename = fold + '/' + 'Jij' + '_avg.csv'
    if not file_exists(filename):
        np.savetxt(filename, J, delimiter=default_delimiter, fmt=format)

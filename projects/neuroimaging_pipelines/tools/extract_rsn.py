import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import nibabel as nib
from generalize_ising_model.ising_utils import makedir
import scipy.ndimage as ndim

parcellation_rsn_path = '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn/Parcels_MNI_222.nii'
parcellation_rsn_labels_path = '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn/Parcels.csv'
output_path = '/home/brainlab/Desktop/rsn_parcellations/'

n_groups = [2,3,4,5,6,7,8,9,10]

img = nib.load(parcellation_rsn_path)
data_img = img.get_data()
affine_img = img.affine

data = pd.read_csv(parcellation_rsn_labels_path)
network_list = data['Community'].unique()

for network_name in network_list:
    network = data.loc[data['Community'] == network_name]

    print('Extracting groups from: ' + network_name)
    for num in n_groups:
        print('Number of groups: ' + str(num))
        network_centroids = network['Centroid (MNI)'].tolist()

        centroid_list_splited = [[np.float(value) for value in centroid.split(' ')] for centroid in network_centroids]
        network_centroids_nd = np.asarray(centroid_list_splited)

        if num <= network_centroids_nd.shape[0]:
            kmeans = KMeans(n_clusters=num)
            kmeans.fit(network_centroids_nd)

            y_kmeans = kmeans.predict(network_centroids_nd)

            indexs = network['ParcelID'].tolist()

            new_img_data = np.zeros(data_img.shape)

            for index, group in zip(indexs, y_kmeans):
                temporal_img_data = np.zeros(data_img.shape)
                temporal_img_data[np.where(data_img == index)] = 1

                temporal_img_data = ndim.binary_fill_holes(temporal_img_data)
                temporal_img_data = ndim.binary_dilation(temporal_img_data)
                temporal_img_data = ndim.binary_fill_holes(temporal_img_data)

                new_img_data[temporal_img_data] = group + 1

            makedir(output_path + network_name + '/')
            nib.save(nib.Nifti1Image(new_img_data.astype(np.float32), affine_img), output_path + network_name + '/' + network_name + '_parcellation_' + str(num) + '.nii')
import os
from natsort import natsorted
from generalize_ising_model.ising_utils import makedir
from dipy.io.streamline import save_trk, load_trk
import nibabel as nib
from dipy.tracking.utils import connectivity_matrix
import matplotlib
import numpy as np
from nilearn import plotting

parcellations_path = '/home/brainlab/Desktop/rsn_parcellations/'
subjects_path = '/home/brainlab/Desktop/Rudas/Data/testkokokoko/subjects/'
output_path = '/home/brainlab/Desktop/Rudas/Data/testkokokoko/output/'

for subject in natsorted(os.listdir(subjects_path)):
    print(subject)
    makedir(output_path + subject)
    trk_file = nib.streamlines.load(subjects_path + subject + '/' + 'tractography.trk', lazy_load=True)

    for network in natsorted(os.listdir(parcellations_path)):
        print(network)
        makedir(output_path + subject + '/' + network)
        for size in natsorted(os.listdir(parcellations_path + network)):
            print(size)
            makedir(output_path + subject + '/' + network + '/' + size)
            parcellation_img = nib.load(parcellations_path + network + '/' + size)
            parcellation_data = parcellation_img.get_data()
            parcellation_affine = parcellation_img.affine

            M, grouping = connectivity_matrix(trk_file.streamlines, parcellation_data, affine=trk_file.affine,
                                              return_mapping=True,
                                              mapping_as_streamlines=True)

            M = M[1:, 1:]  # Removing firsts column and row (Index = 0 is the background)
            M[range(M.shape[0]), range(M.shape[0])] = 0  # Removing element over the diagonal

            np.savetxt('Jij.csv', M, delimiter=',', fmt='%d')

            fig, ax = matplotlib.pyplot.subplots()
            plotting.plot_matrix(M, colorbar=True, figure=fig)
            fig.savefig('Jij.png', dpi=1200)

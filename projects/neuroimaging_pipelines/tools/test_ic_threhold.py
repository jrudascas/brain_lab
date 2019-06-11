from nilearn.image import iter_img
import nibabel as nib
from nilearn.plotting import plot_stat_map
import numpy as np
import matplotlib.pyplot as plt

path_ic = '/home/brainlab/Desktop/testcristian/fmri/output/datasink/preprocessing/sub-sub1/descomposition_canica.nii.gz'
number_threshold = 10

for i, cur_img in enumerate(iter_img(nib.load(path_ic))):
    print('ica_ic_' + str(i + 1))

    max = np.max(cur_img.get_data())
    range = np.linspace(0, max, number_threshold)
    cont = 1
    f = plt.figure(figsize=(18, 10))
    for threshold in range:
        ax = f.add_subplot(5, 2, cont)
        print('threshold: ' + str(threshold))
        plot_stat_map(cur_img, axes=ax, display_mode="ortho", colorbar=True, threshold=threshold)
        cont +=1
    f.show()

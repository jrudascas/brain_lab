import numpy as np
import nibabel as nib
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_stat_map

from neuroimaging_pipelines.tools.compute_isc import (isc, bootstrap_isc, load_images,
                                                      load_boolean_mask, mask_images,
                                                      MaskedMultiSubjectData)

def run(roo_path, path_output):

    name_cleaned = '/fmri_cleaned.nii'

    path = [root_path + '2014_05_02_02CB' + name_cleaned,
            root_path + '2014_05_16_16RA' + name_cleaned,
            root_path + '2014_05_30_30AQ' + name_cleaned,
            root_path + '2014_07_04_04HD' + name_cleaned,
            root_path + '2014_07_04_04SG' + name_cleaned,
            root_path + '2014_08_13_13CA' + name_cleaned,
            root_path + '2014_10_08_08BC' + name_cleaned,
            root_path + '2014_10_08_08VR' + name_cleaned,
            root_path + '2014_10_22_22CY' + name_cleaned,##
            root_path + '2014_10_22_22TK' + name_cleaned,
            root_path + '2014_11_17_17EK' + name_cleaned,
            root_path + '2014_11_17_17NA' + name_cleaned,
            root_path + '2014_11_19_19SA' + name_cleaned,
            root_path + '2014_11_19_AK' + name_cleaned,
            root_path + '2014_11_25.25JK' + name_cleaned,
            root_path + '2014_11_27_27HF' + name_cleaned,
            root_path + '2014_12_10_10JR' + name_cleaned]

    # path = ['/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/fmri_cleaned.nii',
    #        '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_16_16RA/fmri_cleaned.nii',
    #        '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_30_30AQ/fmri_cleaned.nii']
    #        '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_07_04_04HD/fmri_cleaned.nii',]

    mask = '/home/brainlab/Desktop/Rudas/Data/Propofol/MNI152_T1_2mm_brain_mask.nii.gz'

    ref_nii = nib.load(mask)

    mask_img = load_boolean_mask(mask)
    mask_coords = np.where(mask_img)

    func_imgs = load_images(path)
    masked_imgs = mask_images(func_imgs, mask_img)

    orig_data = MaskedMultiSubjectData.from_masked_images(masked_imgs, len(path))

    print("Original fMRI data shape:", orig_data.shape)

    # Trim off non-story TRs and 12 s post-onset
    data = orig_data

    # Z-score time series for each voxel
    data = zscore(data, axis=0)

    # Leave-one-out approach
    iscs = isc(data, pairwise=False, tolerate_nans=.8)

    # Run bootstrap hypothesis test on ISCs

    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                  ci_percentile=95,
                                                  summary_statistic='median',
                                                  n_bootstraps=1000)

    # Get voxels without NaNs
    nonnan_mask = ~np.isnan(observed)
    nonnan_coords = np.where(nonnan_mask)

    # Mask both the ISC and p-value map to exclude NaNs
    nonnan_isc = observed[nonnan_mask]
    nonnan_p = p[nonnan_mask]

    # Get FDR-controlled q-values
    threshold = .05

    nonnan_q = multipletests(nonnan_p, alpha=threshold, method='fdr_bh')[1]

    print(str(np.sum(nonnan_q < threshold)) + " significant voxels controlling FDR at " + str(threshold))


    # Threshold ISCs according FDR-controlled threshold
    nonnan_isc[nonnan_q >= threshold] = np.nan

    # Reinsert thresholded ISCs back into whole brain image
    isc_thresh = np.full(observed.shape, np.nan)
    isc_thresh[nonnan_coords] = nonnan_isc

    # Create empty 3D image and populate
    # with thresholded ISC values
    isc_img = np.full(ref_nii.shape, np.nan)
    isc_img[mask_coords] = isc_thresh



    display = plot_stat_map(
        nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header),
        cmap='RdYlBu_r',
        cut_coords=(-61, -20, 8))

    display.savefig(path_output + '_1.png')
    display.close()

    plot_stat_map(
        nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header),
        cmap='RdYlBu_r',
        cut_coords=(0, -65, 40))
    display.savefig(path_output + '_2.png')
    display.close()



#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Resting/output/datasink/preprocessing/sub-'
#run(root_path, 'awake_resting')
#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-'
#run(root_path, 'awake_task')

#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Deep/Resting/output/datasink/preprocessing/sub-'
#run(root_path, 'deep_resting')
#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Deep/Task/output/datasink/preprocessing/sub-'
#run(root_path, 'deep_task')

#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Mild/Resting/output/datasink/preprocessing/sub-'
#run(root_path, 'mild_resting')
#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Mild/Task/output/datasink/preprocessing/sub-'
#run(root_path, 'mild_taskg')

#root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Recovery/Resting/output/datasink/preprocessing/sub-'
#run(root_path, 'recovery_resting')
root_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Recovery/Task/output/datasink/preprocessing/sub-'
run(root_path, 'recovery_task')

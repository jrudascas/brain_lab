from projects.neuroimaging_pipelines.pipeline import PipelinefMRI

paths = {'input_path': '/home/jrudascas/Desktop/Tesis/data/data_test',
         'template_spm_path': '/home/jrudascas/Desktop/Tesis/data/spm/spm12/tpm/TPM.nii',
         'mcr_path': '/opt/mcr/v95',
         'spm_path': '/home/jrudascas/Desktop/Tesis/data/spm/spm12_r7487_Linux_R2018b/spm12/run_spm12.sh',
         'image_parcellation_path': ['/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels-19cwpgu/Parcels_MNI_222.nii'],
         'labels_parcellation_path': None,
         't1_relative_path':'t1.nii',
         'fmri_relative_path':'fmri.nii',
         'mask_mni_path': '/home/jrudascas/Desktop/Tesis/data/parcellations/MNI152_T1_2mm_brain_mask.nii.gz'}

parameters = {'fwhm': 8,
              'tr': 2,
              'init_volume': 6,
              'iso_size': 2,
              'low_pass': 0.1,
              'high_pass': 0.01}

subject_list = ['sub_1']

pipeline = PipelinefMRI(paths=paths, parameters=parameters, subject_list=subject_list)
pipeline.run()

from neuroimaging_pipelines.pipeline import PipelineDWI

paths = {'input_path': '/home/brainlab/Desktop/Rudas/Data/dwitest/HCP/parcellation4',
         'image_parcellation_path': ['/home/brainlab/Desktop/Rudas/Data/dwitest/HCP/parcellation4/atlas_NMI_2mm.nii'],
         'reference': '/home/brainlab/Desktop/Rudas/Data/Parcellation/MNI152_T1_2mm_brain.nii.gz',
         'template_spm_path': '/home/brainlab/Desktop/Rudas/Data/Parcellation/TPM.nii',
         'mcr_path': '/home/brainlab/Desktop/Rudas/Tools/MCR/v713',
         'spm_path': '/home/brainlab/Desktop/Rudas/Tools/spm12_r7487/spm12/run_spm12.sh'}

parameters = {'iso_size': 2}

subject_list = ['sub1']

pipeline = PipelineDWI(paths=paths, parameters=parameters, subject_list=subject_list)
pipeline.run()



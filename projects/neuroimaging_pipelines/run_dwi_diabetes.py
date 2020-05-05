from projects.neuroimaging_pipelines.pipeline import PipelineDWI

paths = {'input_path': '/home/brainlab/Desktop/Nichols/DMR_MRIdata',
         'image_parcellation_path': [
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/Auditory/Auditory_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/CinguloOperc/CinguloOperc_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/CinguloParietal/CinguloParietal_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/Default/Default_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/DorsalAttn/DorsalAttn_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/FrontoParietal/FrontoParietal_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/RetrosplenialTemporal/RetrosplenialTemporal_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/SMhand/SMhand_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/SMmouth/SMmouth_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/VentralAttn/VentralAttn_parcellation_5.nii',
             '/home/brainlab/Desktop/Rudas/Data/Parcellation/rsn_parcellations/Visual/Visual_parcellation_5.nii'],
         'reference': '/home/brainlab/Desktop/Rudas/Data/Parcellation/MNI152_T1_2mm_brain.nii.gz',
         'template_spm_path': '/home/brainlab/Desktop/Rudas/Data/Parcellation/TPM.nii',
         'mcr_path': '/home/brainlab/Desktop/Rudas/Tools/MCR/v713',
         'spm_path': '/home/brainlab/Desktop/Rudas/Tools/spm12_r7487/spm12/run_spm12.sh',
         't1_relative_path': 'data.nii',
         'dwi_relative_path': 'dwi/dwi_data.nii.gz',
         'bvec_relative_path': 'dwi/bvec',
         'bval_relative_path': 'dwi/bval'}

parameters = {'iso_size': 1.5}

subject_list = ['c001'
                'c002',
                'c005',
                'c006',
                'c007',
                'c008',
                'c009',
                'c010',
                'c012',
                'c013',
                'c015',
                'c016',
                'c017',
                'c018',
                'c019',
                's002',
                's005',
                's006',
                's009',
                's011',
                's017',
                's018',
                's019',
                's020',
                'si006',
                'si009']

pipeline = PipelineDWI(paths=paths, parameters=parameters, subject_list=subject_list)
pipeline.run()

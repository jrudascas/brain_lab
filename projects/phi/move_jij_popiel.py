from projects.phi.tools.utils import *
import numpy as np

main_in = '/home/brainlab/Desktop/Rudas/Data/hcp_100/output/workingdir/preproc/'
main_out = '/home/brainlab/Desktop/Popiel/HCP_Jij/'


sub_list = ['_subject_id_211720_3T_Diffusion_preproc/',
            '_subject_id_212318_3T_Diffusion_preproc/',
            '_subject_id_214423_3T_Diffusion_preproc/',
            '_subject_id_221319_3T_Diffusion_preproc/',
            '_subject_id_239944_3T_Diffusion_preproc/',
            '_subject_id_245333_3T_Diffusion_preproc/',
            '_subject_id_280739_3T_Diffusion_preproc/',
            '_subject_id_298051_3T_Diffusion_preproc/',
            '_subject_id_366446_3T_Diffusion_preproc/',
            '_subject_id_397760_3T_Diffusion_preproc/',
            '_subject_id_414229_3T_Diffusion_preproc/',
            '_subject_id_499566_3T_Diffusion_preproc/',
            '_subject_id_654754_3T_Diffusion_preproc/',
            '_subject_id_672756_3T_Diffusion_preproc/',
            '_subject_id_751348_3T_Diffusion_preproc/',
            '_subject_id_756055_3T_Diffusion_preproc/',
            '_subject_id_792564_3T_Diffusion_preproc/',
            '_subject_id_856766_3T_Diffusion_preproc/',
            '_subject_id_857263_3T_Diffusion_preproc/',
            '_subject_id_899885_3T_Diffusion_preproc/']

list_of_parcellations = ['_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Auditory..Auditory_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..CinguloOperc..CinguloOperc_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..CinguloParietal..CinguloParietal_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Default..Default_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..DorsalAttn..DorsalAttn_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..FrontoParietal..FrontoParietal_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..RetrosplenialTemporal..RetrosplenialTemporal_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..SMhand..SMhand_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..SMmouth..SMmouth_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..VentralAttn..VentralAttn_parcellation_5.nii/tractography/Jij.csv',
                         '_atlas_to_apply_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Visual..Visual_parcellation_5.nii/tractography/Jij.csv']

new_parc_names = ['Aud', 'CinguloOperc','CinguloParietal','DMN','Dorsal','FrontoParietal','Retrosplenial','SMhand','SMmouth','Ventral','Visual']



count = 0
for subject in sub_list:
    count += 1
    for i in range(len(list_of_parcellations)):
        load_path = main_in + subject + list_of_parcellations[i]
        J = load_matrix(load_path)

        save_path_main = main_out + 'sub' + str(count)

        if makedir2(save_path_main):
            save_path = save_path_main + '/' + new_parc_names[i]
            if makedir2(save_path):
                default_delimiter = ','
                format = '%1.5f'
                filename1 = save_path + '/Jij.csv'

                if not file_exists(filename1):
                    np.savetxt(filename1 , np.asarray(J), delimiter=default_delimiter, fmt=format)

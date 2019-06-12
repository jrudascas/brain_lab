from projects.phi.tools.utils import *
import os
import numpy as np

ts_name = 'time_series.csv'
rsn_path = [
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Auditory..Auditory_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..CinguloOperc..CinguloOperc_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Default..Default_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..DorsalAttn..DorsalAttn_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..FrontoParietal..FrontoParietal_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..RetrosplenialTemporal..RetrosplenialTemporal_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..SMhand..SMhand_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..SMmouth..SMmouth_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..VentralAttn..VentralAttn_parcellation_5.nii',
    '/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Visual..Visual_parcellation_5.nii']
number_regions = 5
brain_states = ['Awake']#,'Deep','Mild','Recovery']

for state in brain_states:

    test_path = '/home/user/Desktop/data_phi/Propofol/' + state + '/'
    test_path_pre = test_path + 'datasink/preprocessing/'


    sub_num = 1
    for subfolder in os.listdir(test_path_pre):
        sub_path = test_path_pre  + subfolder
        for folder in rsn_path:
            filepath = sub_path + folder + '/' + ts_name
            ts = load_matrix(filepath)
            timeSeries = ts[:, 0:number_regions].astype(np.float32)
            ts_path = test_path + filepath.split('.')[-3]
            save_ts(timeSeries,ts_path,filepath,sub_num)
        sub_num+=1



    new_path_list = ['/home/user/Desktop/data_phi/Propofol/' + state + '/Auditory_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/CinguloOperc_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/Default_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/DorsalAttn_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/FrontoParietal_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/RetrosplenialTemporal_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMhand_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMmouth_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/VentralAttn_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/Visual_parcellation_5', ]

    for fold in new_path_list:
        ts = make_ts_array(fold)
        empirical_tpm_concat(ts,fold)


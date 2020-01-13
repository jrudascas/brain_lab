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
brain_states = ['Awake','Deep','Mild','Recovery']
                # ['Recovery']


for state in brain_states:

    test_path = '/home/user/Desktop/data_phi/Propofol/task/' + state + '/'


    sub_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    for folder in rsn_path:
        for sub_num in sub_nums:
            sub_path = test_path + 'Sub' + str(sub_num)
            filepath = sub_path + folder + '/' + ts_name
            ts = load_matrix(filepath)
            timeSeries = ts[:, 0:number_regions].astype(np.float32)
            ts_path = test_path + folder.split('.')[-2] + '/SbyS'
            save_ts(timeSeries, ts_path, filepath, sub_num)




    new_path_list = ['/home/user/Desktop/data_phi/Propofol/task/' + state + '/Auditory_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/CinguloOperc_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/Default_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/DorsalAttn_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/FrontoParietal_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/RetrosplenialTemporal_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/SMhand_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/SMmouth_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/VentralAttn_parcellation_5/SbyS',
                    '/home/user/Desktop/data_phi/Propofol/task/' + state + '/Visual_parcellation_5/SbyS']




    for fold in new_path_list:
        ts = make_ts_array2(fold)
        empirical_tpm_concat_sbys_2(ts,fold)
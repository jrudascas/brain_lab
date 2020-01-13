from projects.phi.tools.utils import *
import os
import numpy as np

test_path = '/home/user/Desktop/data_phi/Propofol/Awake/datasink/preprocessing/'
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
tpmList = []
number_regions = 5

sub_num = 1
for subfolder in os.listdir(test_path):
    sub_path = test_path + '/' + subfolder
    for folder in rsn_path:
        filepath = sub_path + folder + '/' + ts_name
        ts = load_matrix(filepath)
        timeSeries = ts[:, 0:number_regions].astype(np.float32)
        ts_path = test_path + filepath.split('.')[-3]
        save_ts(timeSeries,ts_path,filepath,sub_num)
    sub_num+=1



'''
for i in range(4):
    subject_folder = '/Sub' + str(i + 1) + '/'
    filepath = test_path + subject_folder + ts_name
    t = load_matrix(filepath)
    number_regions = 5
    timeSeries = t[:, 0:number_regions].astype(np.float32)
    # tpmList.append(empirical_tpm_eps(timeSeries))
    tpmList.append(empirical_tpm_og(timeSeries))

ts = make_ts_array(test_path,ts_name)

empirical_tpm_concat(ts)

'''

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


for state in brain_states:

    new_path_list = ['/home/user/Desktop/data_phi/Propofol/' + state + '/Auditory_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/CinguloOperc_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/Default_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/DorsalAttn_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/FrontoParietal_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/RetrosplenialTemporal_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMhand_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/SMmouth_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/VentralAttn_parcellation_5',
                    '/home/user/Desktop/data_phi/Propofol/' + state + '/Visual_parcellation_5']




    for fold in new_path_list:

        save_path = fold + '/SbyS/'
        makedir2(save_path)
        ts = make_ts_array(fold)
        empirical_tpm_concat_sbys(ts,save_path)
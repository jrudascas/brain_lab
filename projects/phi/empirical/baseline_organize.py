from projects.phi.tools.utils import *
import os
import numpy as np

ts_name = 'time_series.csv'
brain_states = ['Awake','Deep','Mild','Recovery']
                # ['Recovery']


for state in brain_states:

    big_matrix_list = []

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
        ts = make_ts_array2(fold)
        long_ts = empirical_ts_concat(ts,all=True)
        big_matrix_list.append(long_ts)



    ts_array = np.squeeze(np.array(big_matrix_list)).reshape((4165,50))
    ts_path = '/home/user/Desktop/data_phi/Propofol/Baseline/' + state + '/'
    save_ts_all(ts_array,ts_path)


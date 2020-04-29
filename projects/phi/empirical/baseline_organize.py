from projects.phi.tools.utils import *
import os
import numpy as np

ts_name = 'time_series.csv'
brain_states = ['Awake','Deep','Mild','Recovery']
                # ['Recovery']


for state in brain_states:

    big_matrix_list = []

    new_path_list = ['/home/user/Desktop/data_phi/Propofol/task/' + state + '/Auditory_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/CinguloOperc_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/Default_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/DorsalAttn_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/FrontoParietal_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/RetrosplenialTemporal_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/SMhand_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/SMmouth_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/VentralAttn_parcellation_5',
                     '/home/user/Desktop/data_phi/Propofol/task/' + state + '/Visual_parcellation_5']


    for fold in new_path_list:
        ts = make_ts_array2(fold)
        long_ts = empirical_ts_concat(ts,all=True)
        big_matrix_list.append(long_ts)
    b = []
    print(state)
    for i in range(len(big_matrix_list)):
        b2 = big_matrix_list[i]

        if big_matrix_list[i].shape[0] >= 2352:
            b2= np.delete(big_matrix_list[i],np.arange(2352,b2.shape[0]),0)

        print(b2.shape)
        b.append(b2)

    ts_array = np.squeeze(np.array(b)).reshape((2352,50))
    ts_path = '/home/user/Desktop/data_phi/Propofol/task/Baseline/' + state + '/'
    save_ts_all(ts_array,ts_path)

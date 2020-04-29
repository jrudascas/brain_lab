import numpy as np
from projects.qm_brain.utils.utils import *


'''
Pseudo code

load in time series for each subject

for each electrode
    for each subject
        for each other subject
            calculate correlation coefficienct
            
'''

stimuli =[('New Data/',15),('BYD/',14),('nyc_data/',13)]

titles = ['Taken Phase','Taken Scrambled Phase','BYD Phase','BYD Scrambled Phase','Rest Phase','Present Phase']

count=0

for stimulus in stimuli:

    main_path = '/home/user/Desktop/QMBrain/' + stimulus[0]

    x = load_matrix(main_path + 'x_chanloc.csv')
    y = load_matrix(main_path + 'y_chanloc.csv')


    condition_list = ['Cond10/','Cond12/']

    num_sub = stimulus[1]
    num_electrodes = 92

    #list_sub = np.arange(num_sub)

    const =  (num_sub**2 - num_sub)/2

    for condition in condition_list:

        data_list = []
        phase_list = []
        prob_list = []
        save_path = main_path + condition +'/'

        for i in range(num_sub):

            data_path = main_path + condition + str(i + 1) + '/data.csv'

            phase,normAmp, prob = process_eeg_data(np.squeeze(load_matrix(data_path)))

            phase_list.append(phase)
            prob_list.append(prob)
            data_list.append(np.squeeze(load_matrix(data_path)))


        time_series_isc = np.array(data_list)
        time_series_prob = np.array(prob_list)
        time_series_phase = np.array(phase_list)

        #print(time_series.shape)

        num_electrodes = time_series_isc.shape[2]
        num_time_points = time_series_isc.shape[1]

        rr_isc = np.zeros((num_sub,num_sub,num_electrodes))
        r_bar_isc = np.zeros((num_electrodes))

        rr_prob = np.zeros((num_sub,num_sub,num_electrodes))
        r_bar_prob = np.zeros((num_electrodes))

        rr_phase = np.zeros((num_sub,num_sub,num_electrodes))
        r_bar_phase = np.zeros((num_electrodes))

        for elec in range(num_electrodes):
            for sub1 in range(num_sub):
                for sub2 in range(num_sub):
                    if sub1>sub2:
                        r_isc = np.corrcoef(time_series_isc[sub1,:,elec],time_series_isc[sub2,:,elec])
                        r_prob = np.corrcoef(time_series_prob[sub1, :, elec], time_series_prob[sub2, :, elec])
                        r_phase = np.corrcoef(time_series_phase[sub1, :, elec], time_series_phase[sub2, :, elec])

                        #print(r)

                        rr_isc[sub1,sub2,elec] = r_isc[0,1]
                        rr_prob[sub1, sub2, elec] = r_prob[0, 1]
                        rr_phase[sub1, sub2, elec] = r_phase[0, 1]

            r_bar_isc[elec] = np.sum(rr_isc[:,:,elec]) / const
            r_bar_prob[elec] = np.sum(rr_prob[:,:,elec]) / const
            r_bar_phase[elec] = np.sum(rr_phase[:,:,elec]) / const
        #save_file(rr,save_path,'isc_electrodes_phase')


        save_file(r_bar_isc,save_path,'avg_isc')
        save_file(r_bar_prob,save_path,'avg_isc_prob')
        save_file(r_bar_phase,save_path,'avg_isc_phase')





# 2/(num_electrodes**2 - num_electrodes)
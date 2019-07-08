from projects.qm_brain.utils import *
import numpy as np
import matplotlib.pyplot as plt

num_subjects = 15
main_path = '/home/user/Desktop/QMBrain/New Data/'

condition_list = ['Cond10/','Cond12/']


for condition in condition_list:

    for i in range(num_subjects):

        subject_path = main_path + condition + str(i + 1) + '/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        times = probability.shape[0]


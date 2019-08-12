from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd

main_path = '/home/user/Desktop/QMBrain/'
condition_list = ['Cond10/','Cond12/']
exp_list = ['New Data/','BYD/','RestData/']


save_path = '/home/user/Desktop/QMBrain/tpm/8_regions/'

hist_exp = []

for exp in exp_list:
    if exp == 'New Data/':
        num_subjects = 15
    elif exp == 'BYD/':
        num_subjects = 14
    elif exp == 'RestData/':
        num_subjects = 13
        condition_list = ['/']
    else:
        num_subjects = []

    dictX, dictY = {}, {}

    for condition in condition_list:

        min_x_list, min_y_list, tot_list = [], [], []

        max_x, max_y = [], []

        for i in range(num_subjects):
            subject_path = main_path + condition + str(i + 1) + '/results/'

            uncertain_x = load_matrix(subject_path + 'DeltaXDeltaPX.csv')
            uncertain_y = load_matrix(subject_path + 'DeltaXDeltaPY.csv')

            min_x_list.append(np.min(uncertain_x))
            min_y_list.append(np.min(uncertain_y))


        avg_x = np.mean(np.array(min_x_list))
        avg_y = np.mean(np.array(min_y_list))

        std_x = np.std(np.array(min_x_list))
        std_y = np.std(np.array(min_y_list))

        tot_list.append(min_x_list)
        tot_list.append(min_y_list)

        print('The minimum uncertainty is: ', avg_x, 'plus or minus', np.std(np.array(min_x_list)), condition)
        print('The minimum uncertainty for y is: ', avg_y, 'plus or minus', np.std(np.array(min_y_list)), condition)
        print('The minimum uncertainty is: ', np.mean(np.array(tot_list)), 'plus or minus', np.std(np.array(tot_list)),
              condition)
        print('The maximum uncertainty for x is: ', np.max(np.array(max_x)), condition)
        print('The maximum uncertainty for y is: ', np.max(np.array(max_y)), condition)





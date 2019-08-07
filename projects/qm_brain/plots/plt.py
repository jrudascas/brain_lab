from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt


main_path = '/home/user/Desktop/QMBrain/RestData/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

coord_stack = zip_x_y(x,y)

condition_list = ['/']#'Cond10/','Cond12/']

for condition in condition_list:

    for i in range(13):

        subject_path = main_path + condition + str(i + 1) + '/results/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        if not file_exists(subject_path+'DeltaXDeltaPY.csv'):

            filepathPos = subject_path + 'position_wavefunction_1d.npy'
            filepathMom = subject_path + 'momentum_wavefunction.npy'

